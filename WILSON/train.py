import torch
import numpy as np
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import tqdm
from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss
from torch.cuda import amp
from segmentation_module import make_model, TestAugmentation
import tasks
from torch.nn.parallel import DistributedDataParallel
import os.path as osp
from wss.modules import PAMR, ASPP
from utils.utils import denorm, label_to_one_hot
from wss.single_stage import pseudo_gtmask, balanced_mask_loss_ce, balanced_mask_loss_unce, balanced_mask_loss_ce_bg_mask
from utils.wss_loss import bce_loss, ngwp_focal, binarize, bce_loss_bg_mask
from segmentation_module import get_norm
from utils.scheduler import get_scheduler


class Trainer:
    def __init__(self, logger, device, opts):
        self.logger = logger
        self.device = device
        self.opts = opts
        self.scaler = amp.GradScaler()

        self.classes = classes = tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)
        self.final_step = list(tasks.tasks[opts.dataset][opts.task].keys())[-1]
        self.n_dense_classes = len(tasks.tasks[opts.dataset][opts.task][0])

        if classes is not None:
            new_classes = classes[-1]
            self.tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = self.tot_classes - new_classes
        else:
            self.old_classes = 0

        self.model = make_model(opts, classes=classes)

        if opts.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))
            self.model_old.to(self.device)
            # freeze old model and set eval mode
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        self.weakly = opts.weakly and opts.step > 0
        self.pos_w = opts.pos_w
        self.use_aff = opts.affinity
        self.weak_single_stage_dist = opts.ss_dist
        self.pseudo_epoch = opts.pseudo_ep
        cls_classes = self.tot_classes
        self.pseudolabeler = None

        if self.weakly:
            self.affinity = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12]).to(device)
            for p in self.affinity.parameters():
                p.requires_grad = False
            norm = get_norm(opts)
            channels = 4096 if "wide" in opts.backbone else 2048
            self.pseudolabeler = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               norm(256),
                                               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                               norm(256),
                                               nn.Conv2d(256, cls_classes, kernel_size=1, stride=1))

            self.icarl = opts.icarl

        self.optimizer, self.scheduler = self.get_optimizer(opts)

        self.distribute(opts)

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and self.model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and self.model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and self.model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and self.model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

    def get_optimizer(self, opts):
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters()),
                           'weight_decay': opts.weight_decay})

        params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                       'weight_decay': opts.weight_decay, 'lr': opts.lr*opts.lr_head})
        if self.weakly:
            params.append({"params": filter(lambda p: p.requires_grad, self.pseudolabeler.parameters()),
                           'weight_decay': opts.weight_decay, 'lr': opts.lr_pseudo})

        optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)
        scheduler = get_scheduler(opts, optimizer)

        return optimizer, scheduler

    def distribute(self, opts):
        self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[opts.device_id],
                                             output_device=opts.device_id, find_unused_parameters=False)
        if self.weakly:
            self.pseudolabeler = DistributedDataParallel(self.pseudolabeler.to(self.device), device_ids=[opts.device_id],
                                                         output_device=opts.device_id, find_unused_parameters=False)

    def train(self, cur_epoch, train_loader, print_int=10):
        """Train and return epoch loss"""
        optim = self.optimizer
        scheduler = self.scheduler
        device = self.device
        model = self.model
        criterion = self.criterion
        logger = self.logger

        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        epoch_loss = 0.0
        reg_loss = 0.0
        l_cam_out = 0.0
        l_cam_int = 0.0
        l_seg = 0.0
        l_cls = 0.0
        interval_loss = 0.0

        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        if self.weakly:
            l_seg_new = 0.0
            l_seg_old = 0.0
            l_cam_new_tot = 0.0
            l_loc_tot = 0.0
            lde_tot = 0.0
            l_seg_tot = 0.0
            l_seg_new_tot = 0.0
            l_seg_old_tot = 0.0
            l_cls_tot = 0.0

        train_loader.sampler.set_epoch(cur_epoch)

        if distributed.get_rank() == 0:
            tq = tqdm.tqdm(total=len(train_loader))
            tq.set_description("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
        else:
            tq = None

        model.train()
        for cur_step, (images, labels, l1h) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float)
            images = torch.clamp(images , -3, 3)
            l1h = l1h.to(device, dtype=torch.float)  # this are one_hot
            labels = labels.to(device, dtype=torch.long)

            with amp.autocast():
                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.weakly) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, interpolate=False)

                optim.zero_grad()
                outputs, features = model(images, interpolate=False)

                # xxx BCE / Cross Entropy Loss
                if not self.weakly:
                    outputs = F.interpolate(outputs, size=images.shape[-2:], mode="bilinear", align_corners=False)
                    if not self.icarl_only_dist:
                        loss = criterion(outputs, labels)  # B x H x W
                    else:
                        # ICaRL loss -- unique CE+KD
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                    loss = loss.mean()  # scalar

                    # xxx ICARL DISTILLATION
                    if self.icarl_combined:
                        # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                        n_cl_old = outputs_old.shape[1]
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                        l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                      torch.sigmoid(outputs_old))

                    # xxx ILTSS (distillation on features or logits)
                    if self.lde_flag:
                        lde = self.lde * self.lde_loss(features['body'], features_old['body'])

                    if self.lkd_flag:
                        outputs_old = F.interpolate(outputs_old, size=images.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        # resize new output to remove new logits and keep only the old ones
                        lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

                else:
                    bs = images.shape[0]

                    # generating masks for replay
                    rep_mask = (labels.view(bs, -1).sum(dim=1) != 0.0).to(bool)  # masks out replay images (false for rep, true for voc)
                    rep_bg_mask = torch.zeros_like(outputs.detach())
                    rep_bg_mask[torch.stack([(outputs_old.detach().argmax(dim=1) != 0).to(bool)] * self.tot_classes, dim=1)] = 1
                    rep_bg_mask[rep_mask] = 1
                    if self.opts.mask_post_inp:
                        # some voc images have labels all 0, but l1h not all 0. These images got masked out using mask_pre_inp, but will not be masked out with this
                        inp_mask = (l1h[:, self.old_classes-1:].sum(dim=1) != 0.0).to(bool)  # masks out replay images that do not have new classes (false for old only, true else)
                        rep_bg_mask[inp_mask] = 1

                    # add 1-hot labels for old classes in replay data
                    if self.opts.inpainting_old_od:  # inpaint old classes in all data, use for localizer od loss
                        l1h_inp_od = l1h.clone()
                        for i, old_output in enumerate(outputs_old.detach()):
                            uniques = torch.unique(old_output.argmax(dim=0)).cpu()
                            uniques = uniques[(uniques > 0) & (uniques < self.old_classes)]
                            l1h_inp_od[i][uniques - 1] = 1.0
                            if self.opts.inpainting_old:  # additionally use old class inpaintings for other losses
                                l1h[i][uniques - 1] = 1.0
                    elif self.opts.inpainting_old:  # inpaint old classes in replay data, use for other losses
                        for i, old_output in enumerate(outputs_old.detach()):
                            if not rep_mask[i]:
                                uniques = torch.unique(old_output.argmax(dim=0)).cpu()
                                uniques = uniques[(uniques > 0) & (uniques <= self.old_classes)]
                                l1h[i][uniques - 1] = 1.0

                    self.pseudolabeler.eval()
                    int_masks = self.pseudolabeler(features['body']).detach()

                    self.pseudolabeler.train()
                    int_masks_raw = self.pseudolabeler(features['body'])

                    if self.opts.no_mask | self.opts.inpainting_old_od:
                        if self.opts.mask_replay_l_cam_new:
                            l_cam_new = bce_loss(int_masks_raw[rep_mask], l1h_inp_od[rep_mask], mode=self.opts.cam, reduction='mean')
                        elif self.opts.mask_pre_inp | self.opts.mask_post_inp:
                            l_cam_new = bce_loss_bg_mask(int_masks_raw, l1h_inp_od,
                                                rep_bg_mask, mode=self.opts.cam, reduction='mean')
                        else:
                            l_cam_new = bce_loss(int_masks_raw, l1h_inp_od, mode=self.opts.cam, reduction='mean')
                    else:
                        # assuming replay data with mixed classes and all-0 1-hot labels
                        # this would train model to not see new classes, but they might be present
                        # disable replay
                        if self.opts.mask_replay_l_cam_new:
                            l_cam_new = bce_loss(int_masks_raw[rep_mask], l1h[rep_mask, self.old_classes - 1:],
                                                mode=self.opts.cam, reduction='mean')
                        elif self.opts.mask_pre_inp | self.opts.mask_post_inp:
                            l_cam_new = bce_loss_bg_mask(int_masks_raw, l1h[:, self.old_classes - 1:],
                                                rep_bg_mask, mode=self.opts.cam, reduction='mean')
                        else:
                            l_cam_new = bce_loss(int_masks_raw, l1h[:, self.old_classes - 1:],
                                                mode=self.opts.cam, reduction='mean')

                    # assuming replay data with mixed classes and all-0 1-hot labels
                    # this traines the model to see old classes -> good
                    # but background shift -> only where old model does not see background (for replay)
                    # alternative: disable since localizer should only care about new classes
                    if self.opts.mask_replay_l_loc:
                        l_loc = (F.binary_cross_entropy_with_logits(int_masks_raw[rep_mask, :self.old_classes],
                                                                   torch.sigmoid(outputs_old)[rep_mask],
                                                                   reduction='none',
                                                                   )).mean()
                    elif self.opts.mask_pre_inp | self.opts.mask_post_inp:
                        l_loc = (F.binary_cross_entropy_with_logits(int_masks_raw[:, :self.old_classes],
                                                                   torch.sigmoid(outputs_old),
                                                                   reduction='none',
                                                                   ) * rep_bg_mask[:, :self.old_classes]).mean()
                    else:
                        l_loc = F.binary_cross_entropy_with_logits(int_masks_raw[:, :self.old_classes],
                                                                torch.sigmoid(outputs_old.detach()),
                                                                reduction='mean',
                                                                )
                    l_cam_int = l_cam_new + l_loc

                    if self.lde_flag:
                        # assuming replay data with mixed classes and all-0 1-hot labels
                        # this pushes model back to old model -> would be good for non-background only
                        # but pixels of feature maps not necessarily aligned with model output -> disable replay
                        if self.opts.mask_replay_lde:
                            lde = self.lde * self.lde_loss(features['body'][rep_mask], features_old['body'][rep_mask])
                        else:
                            lde = self.lde * self.lde_loss(features['body'], features_old['body'])

                    l_cam_out = 0 * outputs[0, 0].mean()  # avoid errors due to DDP

                    if cur_epoch >= self.pseudo_epoch:

                        int_masks_orig = int_masks.softmax(dim=1)
                        int_masks_soft = int_masks.softmax(dim=1)

                        if self.use_aff:
                            image_raw = denorm(images)
                            im = F.interpolate(image_raw, int_masks.shape[-2:], mode="bilinear",
                                               align_corners=True)
                            int_masks_soft = self.affinity(im, int_masks_soft.detach())

                        int_masks_orig[:, 1:] *= l1h[:, :, None, None]
                        int_masks_soft[:, 1:] *= l1h[:, :, None, None]

                        pseudo_gt_seg = pseudo_gtmask(int_masks_soft, ambiguous=True, cutoff_top=0.6,
                                                      cutoff_bkg=0.7, cutoff_low=0.2).detach()  # B x C x HW

                        pseudo_gt_seg_lx = binarize(int_masks_orig)
                        pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + \
                                           ((1-self.opts.alpha) * int_masks_orig)

                        # ignore_mask = (pseudo_gt_seg.sum(1) > 0)
                        px_cls_per_image = pseudo_gt_seg_lx.view(bs, self.tot_classes, -1).sum(dim=-1)
                        # batch_weight = (bs, n_classes), 1 if localizer and l1h agree on existence/absence of new class in image, 0 else
                        batch_weight = torch.eq((px_cls_per_image[:, self.old_classes:] > 0),
                                                l1h[:, self.old_classes - 1:].bool())
                        # batch_weight = (bs), 1 if localizer and l1h agree on existence/absence of all new classes in image, 0 else
                        batch_weight = (
                                    batch_weight.sum(dim=1) == (self.tot_classes - self.old_classes)).float()

                        target_old = torch.sigmoid(outputs_old.detach())

                        target = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                        if self.opts.icarl_bkg == -1:
                            target[:, 0] = torch.min(target[:, 0], pseudo_gt_seg_lx[:, 0])
                        else:
                            target[:, 0] = (1-self.opts.icarl_bkg) * target[:, 0] + \
                                           self.opts.icarl_bkg * pseudo_gt_seg_lx[:, 0]

                        # assuming replay data with mixed classes and all-0 1-hot labels
                        # for old classes, this trains model to find old classes -> good, but background shift -> only non-background (for replay)
                        # for new classes, this trains model to not see new classes -> disable replay
                        if self.opts.mask_replay_l_seg:
                            l_seg_new = F.binary_cross_entropy_with_logits(outputs[rep_mask, self.old_classes:], target[rep_mask, self.old_classes:], reduction='none').sum(dim=1)
                            l_seg_old = (F.binary_cross_entropy_with_logits(outputs[:, :self.old_classes], target[:, :self.old_classes], reduction='none') * rep_bg_mask).sum(dim=1)
                            l_seg_new = l_seg_new.view(rep_mask.sum(), -1).mean(dim=-1)
                            l_seg_old = l_seg_old.view(bs, -1).mean(dim=-1)
                            l_seg_new = self.opts.l_seg * (batch_weight[rep_mask] * l_seg_new).sum() / (batch_weight[rep_mask].sum() + 1e-5)
                            l_seg_old = self.opts.l_seg * l_seg_old.sum() / (bs + 1e-5)
                            l_seg = l_seg_new + l_seg_old
                        elif self.opts.mask_pre_inp | self.opts.mask_post_inp:
                            l_seg_new = F.binary_cross_entropy_with_logits(outputs[:, self.old_classes:], target[:, self.old_classes:], reduction='none')
                            l_seg_old = F.binary_cross_entropy_with_logits(outputs[:, :self.old_classes], target[:, :self.old_classes], reduction='none')
                            l_seg_new = (l_seg_new * rep_bg_mask[:, self.old_classes:]).sum(dim=1)
                            l_seg_old = (l_seg_old * rep_bg_mask[:, :self.old_classes]).sum(dim=1)
                            l_seg_new = l_seg_new.view(bs, -1).mean(dim=-1)
                            l_seg_old = l_seg_old.view(bs, -1).mean(dim=-1)
                            l_seg_new = self.opts.l_seg * (batch_weight * l_seg_new).sum() / (batch_weight.sum() + 1e-5)
                            l_seg_old = self.opts.l_seg * (batch_weight * l_seg_old).sum() / (batch_weight.sum() + 1e-5)
                            l_seg = l_seg_new + l_seg_old
                        else:
                            # divide into old $ new to allow for comparison
                            # l_seg = F.binary_cross_entropy_with_logits(outputs, target, reduction='none').sum(dim=1)
                            # l_seg = l_seg.view(bs, -1).mean(dim=-1)
                            # l_seg = self.opts.l_seg * (batch_weight * l_seg).sum() / (batch_weight.sum() + 1e-5)
                            l_seg_new = F.binary_cross_entropy_with_logits(outputs[:, self.old_classes:], target[:, self.old_classes:], reduction='none').sum(dim=1)
                            l_seg_old = F.binary_cross_entropy_with_logits(outputs[:, :self.old_classes], target[:, :self.old_classes], reduction='none').sum(dim=1)
                            l_seg_new = l_seg_new.view(bs, -1).mean(dim=-1)
                            l_seg_old = l_seg_old.view(bs, -1).mean(dim=-1)
                            l_seg_new = self.opts.l_seg * (batch_weight * l_seg_new).sum() / (batch_weight.sum() + 1e-5)
                            l_seg_old = self.opts.l_seg * (batch_weight * l_seg_old).sum() / (batch_weight.sum() + 1e-5)
                            l_seg = l_seg_new + l_seg_old

                        # assuming replay data with mixed classes and all-0 1-hot labels
                        # this trains model to no see new classes -> disable replay
                        if self.opts.mask_replay_l_cls:
                            l_cls = balanced_mask_loss_ce(int_masks_raw[rep_mask], pseudo_gt_seg[rep_mask], l1h[rep_mask])
                        elif self.opts.mask_pre_inp | self.opts.mask_post_inp:
                            l_cls = balanced_mask_loss_ce_bg_mask(int_masks_raw, pseudo_gt_seg, l1h, rep_bg_mask[:, 0])
                        else:
                            l_cls = balanced_mask_loss_ce(int_masks_raw, pseudo_gt_seg, l1h)

                    loss = l_seg + l_cam_out
                    l_reg = l_cls + l_cam_int

                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                loss_tot = loss + lkd + lde + l_icarl + l_reg

            self.scaler.scale(loss_tot).backward()
            self.scaler.step(optim)
            if scheduler is not None:
                scheduler.step()
            self.scaler.update()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if self.weakly:
                l_cam_new_tot += l_cam_new.item() if l_cam_new != 0. else 0.
                l_loc_tot += l_loc.item() if l_loc != 0. else 0.
                lde_tot += lde.item() if lde != 0. else 0.
                l_seg_tot += l_seg.item() if l_seg != 0. else 0.
                l_seg_new_tot += l_seg_new.item() if l_seg_new != 0. else 0.
                l_seg_old_tot += l_seg_old.item() if l_seg_old != 0. else 0.
                l_cls_tot += l_cls.item() if l_cls != 0. else 0.

            if tq is not None:
                tq.update(1)
                tq.set_postfix(loss='%.3f' % loss, l_reg='%.3f' % l_reg)
                # if np.isnan(loss.item()):
                #     raise ValueError("Cls Loss is NaN")
                # if np.isnan(l_reg.item()):
                #     raise ValueError("Reg Loss is NaN")

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.debug(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                             f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss/tot', interval_loss, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_int', l_cam_int, x, intermediate=True)
                    logger.add_scalar('Loss/CAM_out', l_cam_out, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_int', l_cls, x, intermediate=True)
                    logger.add_scalar('Loss/SEG_out', l_seg, x, intermediate=True)
                    logger.commit(intermediate=True)
                interval_loss = 0.0

        if tq is not None:
            tq.close()

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")
        if self.weakly:
            logger.add_scalar("loss_detailed/l_cam_new", l_cam_new_tot / len(train_loader), cur_epoch)
            logger.add_scalar("loss_detailed/l_loc", l_loc_tot / len(train_loader), cur_epoch)
            logger.add_scalar("loss_detailed/lde", lde_tot / len(train_loader), cur_epoch)
            logger.add_scalar("loss_detailed/l_seg", l_seg_tot / len(train_loader), cur_epoch)
            logger.add_scalar("loss_detailed/l_seg_new", l_seg_new_tot / len(train_loader), cur_epoch)
            logger.add_scalar("loss_detailed/l_seg_old", l_seg_old_tot / len(train_loader), cur_epoch)
            logger.add_scalar("loss_detailed/l_cls", l_cls_tot / len(train_loader), cur_epoch)

        return (epoch_loss, reg_loss)

    def inpaint_onehots(self):
        """Knowledge inpainting of image-level pseudolabels for current classes using current model"""
        model = self.model

        import pickle
        import os
        from PIL import Image
        from dataset import transform
        _transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        rep_data_dir = f"{self.opts.replay_root}/{self.opts.task}{'-ov' if self.opts.overlap else ''}"

        model.eval()
        self.pseudolabeler.eval()
        print("Inpainting onehots:")
        if "multistep" in self.opts.replay_root:
            with torch.no_grad():
                for s in range(1, self.opts.step + 1):
                    if s == 1:
                        updated_replay_root = "/".join(self.opts.replay_root.split("/")[:-1]) + "/Base"
                        rep_data_dir_step = f"{updated_replay_root}/{self.opts.task}{'-ov' if self.opts.overlap else ''}"
                    else:
                        rep_data_dir_step = rep_data_dir
                    with open(f"{rep_data_dir_step}/{s - 1}/pseudolabels_1h.pkl", 'rb') as f:
                        onehots = pickle.load(f)
                        for img_name in sorted(os.listdir(f"{rep_data_dir_step}/{s - 1}/images")):
                            img = Image.open(f"{rep_data_dir_step}/{s - 1}/images/{img_name}").convert("RGB")
                            img = _transform(img).to("cuda", dtype=torch.float32).unsqueeze(0)

                            outputs_old, features_old = self.model_old(img, interpolate=False)
                            outputs, features = model(img, interpolate=False)

                            int_masks = self.pseudolabeler(features['body']).detach()
                            int_masks_orig = int_masks.softmax(dim=1)
                            pseudo_gt_seg_lx = binarize(int_masks_orig)
                            pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + ((1-self.opts.alpha) * int_masks_orig)
                            target_old = torch.sigmoid(outputs_old.detach())
                            target = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                            target[:, 0] = torch.min(target[:, 0], pseudo_gt_seg_lx[:, 0])

                            inp_target_old = outputs_old.detach().argmax(dim=1)
                            inp_target_new = outputs.detach().argmax(dim=1)
                            inpainted_labels = torch.zeros_like(inp_target_old)
                            inpainted_labels[inp_target_old > 0] = inp_target_old[inp_target_old > 0]
                            new_spotted_mask = torch.zeros_like(inpainted_labels)
                            new_spotted_mask[(inp_target_old == 0) & (inp_target_new >= self.old_classes) & (inp_target_new == target.argmax(dim=1))] = 1
                            inpainted_labels[new_spotted_mask.to(bool)] = inp_target_new[new_spotted_mask.to(bool)]
                            uniques, counts = torch.unique(inpainted_labels, return_counts=True)
                            num_pixels = np.prod(inpainted_labels.shape)
                            counts = counts[uniques >= self.old_classes]
                            uniques = uniques[uniques >= self.old_classes]
                            for i, unq in enumerate(uniques):
                                if counts[i] >= num_pixels * self.opts.inpainting_threshold:
                                    onehots[f"{img_name[:-4]}.png"][unq - 1] = 1
                    with open(f"{rep_data_dir_step}/{s - 1}/inpainted_pseudolabels_1h.pkl", 'wb') as f:
                        pickle.dump(onehots, f)
        else:
            with torch.no_grad():
                for cl_dir in os.listdir(rep_data_dir):
                    if os.path.isdir(f"{rep_data_dir}/{cl_dir}"):
                        print(f"    {cl_dir}")
                        with open(f"{rep_data_dir}/{cl_dir}/pseudolabels_1h.pkl", 'rb') as f:
                            onehots = pickle.load(f)
                            for img_name in sorted(os.listdir(f"{rep_data_dir}/{cl_dir}/images")):
                                img = Image.open(f"{rep_data_dir}/{cl_dir}/images/{img_name}").convert("RGB")
                                img = _transform(img).to("cuda", dtype=torch.float32).unsqueeze(0)

                                outputs_old, features_old = self.model_old(img, interpolate=False)
                                outputs, features = model(img, interpolate=False)

                                int_masks = self.pseudolabeler(features['body']).detach()
                                int_masks_orig = int_masks.softmax(dim=1)
                                pseudo_gt_seg_lx = binarize(int_masks_orig)
                                pseudo_gt_seg_lx = (self.opts.alpha * pseudo_gt_seg_lx) + ((1-self.opts.alpha) * int_masks_orig)
                                target_old = torch.sigmoid(outputs_old.detach())
                                target = torch.cat((target_old, pseudo_gt_seg_lx[:, self.old_classes:]), dim=1)
                                target[:, 0] = torch.min(target[:, 0], pseudo_gt_seg_lx[:, 0])

                                inp_target_old = outputs_old.detach().argmax(dim=1)
                                inp_target_new = outputs.detach().argmax(dim=1)
                                inpainted_labels = torch.zeros_like(inp_target_old)
                                inpainted_labels[inp_target_old > 0] = inp_target_old[inp_target_old > 0]
                                new_spotted_mask = torch.zeros_like(inpainted_labels)
                                new_spotted_mask[(inp_target_old == 0) & (inp_target_new >= self.old_classes) & (inp_target_new == target.argmax(dim=1))] = 1
                                inpainted_labels[new_spotted_mask.to(bool)] = inp_target_new[new_spotted_mask.to(bool)]
                                uniques, counts = torch.unique(inpainted_labels, return_counts=True)
                                num_pixels = np.prod(inpainted_labels.shape)
                                counts = counts[uniques >= self.old_classes]
                                uniques = uniques[uniques >= self.old_classes]
                                for i, unq in enumerate(uniques):
                                    if counts[i] >= num_pixels * self.opts.inpainting_threshold:
                                        onehots[f"{img_name[:-4]}.png"][unq - 1] = 1
                        with open(f"{rep_data_dir}/{cl_dir}/inpainted_pseudolabels_1h.pkl", 'wb') as f:
                            pickle.dump(onehots, f)
        model.train()
        self.pseudolabeler.train()

    def validate(self, loader, metrics, test=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device

        model.eval()

        with torch.no_grad():
            for x in loader:
                images = x[0].to(device, dtype=torch.float32)
                images = torch.clamp(images, -3, 3)
                labels = x[1].to(device, dtype=torch.long)

                # if self.weakly:
                #     l1h = x[2]

                with amp.autocast():
                    outputs, features = model(images)
                    _, prediction = outputs.max(dim=1)
                # _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy().astype(np.int8)
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            if test and (self.opts.step == self.final_step):
                score = metrics.get_results(final_test=True, n_dense_classes=self.n_dense_classes)
            else:
                score = metrics.get_results()

        return score

    def validate_CAM(self, loader, metrics):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device

        self.pseudolabeler.eval()
        model.eval()

        def classify(images):
            masks = self.pseudolabeler(model(images, as_feature_extractor=True)['body'])
            masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
            masks = masks.softmax(dim=1)
            return masks

        i = -1
        with torch.no_grad():
            for x in tqdm.tqdm(loader):
                i = i+1
                images = x[0].to(device, dtype=torch.float32)
                images = torch.clamp(images, -3, 3)
                labels = x[1].to(device, dtype=torch.long)
                l1h = x[2].to(device, dtype=torch.bool)

                with amp.autocast():
                    masks = classify(images)
                    _, prediction = masks.max(dim=1)
                # _, prediction = masks.max(dim=1)

                labels[labels < self.old_classes] = 0
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy().astype(np.int8)
                metrics.update(labels, prediction)

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

        return score

    def load_step_ckpt(self, path):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            if self.opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                self.model.module.init_new_classifier(self.device)
            # Load state dict from the model state dict, that contains the old model parameters
            new_state = {}
            for k, v in step_checkpoint['model_state'].items():
                new_state[k[7:]] = v
            self.model_old.load_state_dict(new_state, strict=True)  # Load also here old parameters

            self.logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif self.opts.debug:
            self.logger.info(f"[!] WARNING: Unable to find of step {self.opts.step - 1}! "
                             f"Do you really want to do from scratch?")
        else:
            raise FileNotFoundError(path)

    def load_ckpt(self, path):
        opts = self.opts
        assert osp.isfile(path), f"Error, ckpt not found in {path}"

        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.weakly:
            self.pseudolabeler.load_state_dict(checkpoint["pseudolabeler"])

        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        self.logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        del checkpoint

        return cur_epoch, best_score
