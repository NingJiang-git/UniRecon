# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import nibabel as nib
import pandas as pd
from scipy.io import savemat
import random

from torch.nn.parallel import DataParallel, DistributedDataParallel
from define import define_network
from models.base_model import BaseModel
from utils import get_root_logger
import ssl

loss_module = importlib.import_module('losses')
metric_module = importlib.import_module('metrics')
from encoder import ConvEncoderLoss
from discriminator import MultiscaleDiscriminator
from loss import GANLoss, VGGLoss
from losses import TVLoss, Get_gradient
from losses import SSIMLoss
import numpy as np
import cv2
ssl._create_default_https_context = ssl._create_unverified_context

class ImageRestorationModel2(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel2, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.net_dB = MultiscaleDiscriminator()
        self.net_dB = self.model_to_device(self.net_dB)
       
        self.criterionGAN = GANLoss('hinge')
        self.criterionVGG = VGGLoss(0)
        self.criterionFeat = torch.nn.L1Loss()
        self.L1 = torch.nn.L1Loss()

        self.tv = TVLoss(loss_weight=0.1)
        self.get_grad = Get_gradient()
        self.ssim = SSIMLoss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scaler_g = torch.cuda.amp.GradScaler()
        self.scaler_dB = torch.cuda.amp.GradScaler()


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio, 'eps': 1e-3}],
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        dB_optim_params = []
        dB_optim_params_lowlr = []

        for k, v in self.net_dB.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    dB_optim_params_lowlr.append(v)
                else:
                    dB_optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        ratio = 0.1

        self.optimizer_dB = torch.optim.Adam([{'params': dB_optim_params}, {'params': dB_optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio, 'eps': 1e-3}],
                                                **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_dB)
       

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.name = data['gt_name']
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        


    def compute_generator_loss(self):
        G_losses = OrderedDict() #{}
        pred, fake_image = self.net_g(self.lq, self.gt)
   
        pred_fake_B, pred_real_B = self.discriminateB(
            self.gt[:,1,:,:].unsqueeze(dim=1), fake_image, self.lq[:,1,:,:].unsqueeze(dim=1))      
        G_losses['GAN_B'] = 0.1 * self.criterionGAN(pred_fake_B, True, for_discriminator=False)

        num_D = len(pred_fake_B)

        GAN_Feat_loss_B = torch.FloatTensor(1).cuda().fill_(0)
        for i in range(num_D):  # for discriminator
            num_intermediate_outputs = len(pred_fake_B[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                        pred_fake_B[i][j], pred_real_B[i][j].detach())
                GAN_Feat_loss_B += unweighted_loss / num_D
        G_losses['GAN_Feat_B'] = 0.1 * GAN_Feat_loss_B
        G_losses['VGG'] = 0.5 * self.criterionVGG(fake_image.repeat(1,3,1,1), self.lq)

        
        l_pix = 0.
        l_pix += 10 *self.L1(pred, self.gt[:,1,:,:].unsqueeze(dim=1))
        G_losses['l_pix'] = l_pix

        tv = 0.
        tv += self.tv(pred)
        G_losses['tv'] = tv

        l_grad = 0.
        grad_pred = self.get_grad(pred)
        grad_gt = self.get_grad(self.gt[:,1,:,:].unsqueeze(dim=1))
        l_grad += 5 * self.L1(grad_pred, grad_gt)
        G_losses['l_grad'] = l_grad
 

        l_ssim = 0.
        l_ssim += 1.9 *self.ssim(pred, self.gt[:,1,:,:].unsqueeze(dim=1))
        G_losses['l_ssim'] = l_ssim

        return pred, fake_image, G_losses
                                                                                             

    def discriminateB(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.net_dB(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real



    def compute_discriminatorB_loss(self):
        DB_losses = OrderedDict() # {}
        with torch.no_grad():
            _, fake_image = self.net_g(self.lq, self.gt)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminateB(
            self.gt[:,1,:,:].unsqueeze(dim=1), fake_image, self.lq[:,1,:,:].unsqueeze(dim=1))
        DB_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        DB_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        return DB_losses


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast():
            preds, inverse_preds, G_losses = self.compute_generator_loss()
        self.output = preds
        self.output1 = inverse_preds
        l_total = (G_losses['GAN_B'] + G_losses['GAN_Feat_B'])+ G_losses['l_pix'] + G_losses['tv'] + G_losses['l_ssim'] + G_losses['VGG'] + G_losses['l_grad']

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        if torch.any(torch.isnan(l_total)):
            raise ValueError('loss ia nan!!!!!!!')
        self.scaler_g.scale(l_total).backward()

        use_grad_clip = True
        if use_grad_clip:
            self.scaler_g.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()


        self.optimizer_dB.zero_grad()
        DB_losses = self.compute_discriminatorB_loss()
        DB_loss = sum(DB_losses.values()).mean()

        # D_loss.backward()
        self.scaler_dB.scale(DB_loss).backward()
        
        use_grad_clip = True
        if use_grad_clip:        
            self.scaler_dB.unscale_(self.optimizer_dB)
            torch.nn.utils.clip_grad_norm_(self.net_dB.parameters(), 0.01)
        self.scaler_dB.step(self.optimizer_dB)
        self.scaler_dB.update()
        
        self.log_g_dict = G_losses
        self.log_dB_dict = DB_losses



    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred, _ = self.net_g(self.lq[i:j, :, :, :], self.gt[i:j, :, :, :])

                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    # for visualization    
    def get_latest_images(self):
        return [self.lq[0], self.output[0], self.output1[0], self.name[0], self.gt[0]]


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        # dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_1 = {
                metric: []
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = val_data['gt_name'][0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                print('save start!')
                # optional saving
                print('save end!')

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                    self.metric_results_1[name].append(getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_))    

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
        pbar.close()

        current_metric = 0.
        if with_metrics:
            df = pd.DataFrame(self.metric_results_1)
            df.to_csv('./result.csv', index=False, header=True)
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter,
                                      tb_logger):
        log_str = f'Validation,\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:,1,:,:].unsqueeze(dim=1).detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[:,1,:,:].unsqueeze(dim=1).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
