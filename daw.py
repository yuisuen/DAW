import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='DAW: Exploring the Better Weighting Function for Semi-supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=1333, type=int)


class EMA:

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()
        for name, buffer in ema_model.named_buffers():  
            self.shadow[name] = buffer.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buffer in self.model.named_buffers():  
            self.shadow[name] = buffer.clone()

    def update(self, iters):
        ema_decay = min(1 - 1/(iters + 1), self.decay)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - ema_decay) * param.data + ema_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
            
        for name, buffer in self.model.named_buffers():  
            assert name in self.shadow
            new_average_biffer = (1.0 - ema_decay) * buffer.data + ema_decay * self.shadow[name]
            self.shadow[name] = new_average_biffer.clone()
        

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
        for name, buffer in self.model.named_buffers(): 
            if name in self.shadow:
                self.backup[name] = buffer.data
                buffer.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        for name, buffer in self.model.named_buffers(): 
            if name in self.backup:
                buffer.data = self.backup[name]
        self.backup = {}


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    
    model_ema = EMA(model, 0.999)
    model_ema.register()
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']   
    if rank == 0:
        logger.info('Iterations per epoch: {:d}\n'.format(len(trainloader_u)))
    previous_best = 0.0
    previous_best_ema = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    update_thres = optimal_threshold(num_classes=cfg['nclass'], ema_p=0.99, alpha=1.0, \
                                     world_size=world_size)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best_mea: {:.2f}'.format(
                    epoch, optimizer.param_groups[0]['lr'], previous_best_ema))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x, label_ignore_mask),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            label_ignore_mask = label_ignore_mask.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()


            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]    

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.softmax(dim=1).detach()
            
           
            pred_u_w_mix = pred_u_w_mix.softmax(dim=1).detach()
            
            update_thres.update_align(pred_x.softmax(dim=1).detach().clone(), 
                                      torch.cat([pred_u_w,pred_u_w_mix],dim=0),
                                      label_ignore_mask.detach().clone(),
                                      torch.cat([ignore_mask.detach().clone(), \
                                                 ignore_mask_mix.detach().clone()],dim=0)
                                      )
            
            pred_u_w = update_thres.distribution_alignment(pred_u_w)
            pred_u_w_mix = update_thres.distribution_alignment(pred_u_w_mix)

            conf_u_w = pred_u_w.max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
                        
            conf_u_w_mix = pred_u_w_mix.max(dim=1)[0]
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
  

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1] # B*H*W
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            thres_pred = pred_x.detach().clone()
            thres_mask = mask_x.detach().clone()
            loss_x = criterion_l(pred_x, mask_x)
            thresh, safe_thresh, pos_stat, neg_stat = update_thres.update_prob_t(thres_pred, thres_mask, label_ignore_mask.detach().clone())

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= thresh) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= thresh) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            it = epoch * len(trainloader_u) + i
            model_ema.update(iters = it)

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= thresh) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('train/thresh', thresh, iters)
                writer.add_scalar('train/safe_thresh', safe_thresh, iters)
                writer.add_scalar('train/pos_mean', pos_stat[0], iters)
                writer.add_scalar('train/pos_stnd', pos_stat[1], iters)
                writer.add_scalar('train/neg_mean', neg_stat[0], iters)
                writer.add_scalar('train/neg_stnd', neg_stat[1], iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)
        model_ema.apply_shadow()
        mIoU_ema, iou_class_ema = evaluate(model, valloader, eval_mode, cfg)
        model_ema.restore()

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                
            for (cls_idx, iou) in enumerate(iou_class_ema):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU_ema: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU_ema: {:.2f}\n'.format(eval_mode, mIoU_ema))
            
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class_ema):
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        is_best_ema = mIoU_ema > previous_best_ema
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            model_ema.apply_shadow()
            checkpoint_ema = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best_ema,
            }
            model_ema.restore()
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            if is_best_ema:
                torch.save(checkpoint_ema, os.path.join(args.save_path, 'best.pth'))

class optimal_threshold():
    def __init__(self, num_classes=10, ema_p=0.99, alpha=1.0, world_size=None):
        self.num_classes = num_classes
        self.ema_p = ema_p
        self.alpha = alpha
        self.world_size = world_size
        

        self.pos_mean = 1.0 / self.num_classes
        self.pos_std = 1.0 / 2.0
        self.neg_mean = 1.0 / self.num_classes
        self.neg_std = 1.0 / 2.0

        self.pos_var = 1.0 / 2.0
        self.neg_var = 1.0 / 2.0

        # self.thres = 1.0 / self.num_classes
        # self.thres_safe = 1.0 / self.num_classes
        self.thres = 0.95
        self.thres_safe = 0.95
        
        self.lb_prob_t =  torch.ones(self.num_classes,1,1).cuda() / self.num_classes
        self.ulb_prob_t =  torch.ones(self.num_classes,1,1).cuda() / self.num_classes    
    
    @torch.no_grad()
    def distribution_alignment(self, probs):

        probs = probs * self.lb_prob_t.unsqueeze(0) / self.ulb_prob_t.unsqueeze(0)
        probs = probs/probs.sum(dim=1,keepdim = True)
        return probs.detach()
    
    @torch.no_grad()
    def update_align(self,lb_probs,ulb_probs,lb_target,ulb_target):

        lb_max_pred, lb_max_idx = torch.max(lb_probs, dim=1) # B*K*H*W
        ulb_max_pred, ulb_max_idx = torch.max(ulb_probs, dim=1) # B*K*H*W

        valid_lb_max_idx = torch.masked_select(lb_max_idx, lb_target != 255)
        valid_ulb_max_idx = torch.masked_select(ulb_max_idx,  ulb_target != 255)
        lb_batch_classes = torch.unique(valid_lb_max_idx, sorted=True)
        ulb_batch_classes = torch.unique(valid_ulb_max_idx, sorted=True)

        for oi in range(self.num_classes):
            if oi in lb_batch_classes:
                lb_probs = torch.masked_select(lb_max_pred, (lb_max_idx==oi) & (lb_target != 255) )
                self.lb_prob_t[oi,0,0] = self.ema_p*self.lb_prob_t[oi,0,0] +(1-self.ema_p)*torch.mean(lb_probs).cuda()

            if oi in ulb_batch_classes:
                ulb_probs = torch.masked_select(ulb_max_pred, (ulb_max_idx==oi) & (ulb_target != 255) )
                self.ulb_prob_t[oi,0,0] = self.ema_p*self.ulb_prob_t[oi,0,0] +(1-self.ema_p)*torch.mean(ulb_probs).cuda()
     
        
    @torch.no_grad()
    def update_prob_t(self, pred_logits, target, label_ignore_mask):

        pred = F.softmax(pred_logits, dim=1) # B*K*H*W
        max_pred, max_idx = torch.max(pred, dim=1)  # B*H*W

        pos_mask = (max_idx == target) & (label_ignore_mask != 255) # B*H*W
        neg_mask = (max_idx != target) & (label_ignore_mask != 255) # B*H*W

        positive_probs = torch.masked_select(max_pred, pos_mask)
        negative_probs = torch.masked_select(max_pred, neg_mask)
        

        n_p = positive_probs.numel()
        n_n = negative_probs.numel()

        if n_p >= 10:
            mean_pos =  torch.mean(positive_probs).cuda()
        else:
            mean_pos = torch.tensor(self.pos_mean).cuda()

        if n_n >= 10:
            mean_neg =  torch.mean(negative_probs).cuda()
        else:
            mean_neg = torch.tensor(self.neg_mean).cuda() 

        dist.all_reduce(mean_pos)
        dist.all_reduce(mean_neg)
        mean_pos /= self.world_size
        mean_neg /= self.world_size

        if n_p >= 10:
            var_pos = torch.sum((positive_probs - mean_pos)**2) / (n_p - 1 + 1e-8)
            # stnd_pos = torch.sqrt(var_pos).cuda() / 2.0 
        else:
            # stnd_pos = torch.tensor(self.pos_std).cuda()
            var_pos = torch.tensor(self.pos_var).cuda()

        if n_n >= 10:
            var_neg = torch.sum((negative_probs - mean_neg)**2) / (n_n - 1 + 1e-8)
            # stnd_neg = torch.sqrt(var_neg).cuda()
        else:
            # stnd_neg = torch.tensor(self.neg_std).cuda()
            var_neg = torch.tensor(self.neg_var).cuda()

        dist.all_reduce(var_pos)
        dist.all_reduce(var_neg)
        var_pos /= self.world_size
        var_neg /= self.world_size

        mean_pos = self.ema_p * self.pos_mean + (1 - self.ema_p) * mean_pos
        var_pos = self.ema_p * self.pos_var + (1 - self.ema_p) * var_pos
        mean_neg = self.ema_p * self.neg_mean + (1 - self.ema_p) * mean_neg
        var_neg = self.ema_p * self.neg_var + (1 - self.ema_p) * var_neg

        stnd_pos = torch.sqrt(var_pos).cuda()
        stnd_neg = torch.sqrt(var_neg).cuda()

        A = stnd_pos.pow(2) - stnd_neg.pow(2)
        B = 2*((mean_pos*stnd_neg.pow(2)) - (mean_neg*stnd_pos.pow(2)))
        C = (mean_neg*stnd_pos).pow(2) - (mean_pos*stnd_neg).pow(2) + \
            2*(stnd_pos*stnd_neg).pow(2)*torch.log(stnd_neg/(self.alpha*stnd_pos) + 1e-8)

        E = B.pow(2) - 4*A*C
        if E > 0:
            tmp_thres = torch.clamp(((-B + torch.sqrt(E))/(2*A + 1e-10)), min=1.0/self.num_classes, max=1.0).item()
            tmp_thres_safe = torch.clamp((mean_pos-3*stnd_pos), min=1.0/self.num_classes, max=1.0).item()
        else:
            tmp_thres = self.thres
            tmp_thres_safe = self.thres_safe

        self.thres =  tmp_thres
        self.thres_safe = tmp_thres_safe

        self.pos_mean = mean_pos
        self.pos_var = var_pos
        self.pos_std = stnd_pos
        self.neg_mean = mean_neg
        self.neg_var = var_neg
        self.neg_std = stnd_neg
    
        return self.thres, self.thres_safe, (self.pos_mean.item(), self.pos_std.item()), \
            (self.neg_mean.item(), self.neg_std.item())



if __name__ == '__main__':
    main()
