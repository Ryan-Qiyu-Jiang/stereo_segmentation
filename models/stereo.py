import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn

import sys, os
sys.path.append(os.path.abspath("rloss/pytorch/pytorch-deeplab_v3_plus"))
from DenseCRFLoss import DenseCRFLoss
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

sys.path.append(os.path.abspath("monodepth2"))
import networks
from layers import *

from collections import OrderedDict
from utils import denormalizeimage

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = 'cuda'

class StereoProjectionModel(pl.LightningModule):

    def __init__(self, lr=7e-3, batch_size=1, width=640, height=192):
        super().__init__()
        self.num_classes = 9
        self.model = DeepLab(num_classes=self.num_classes) 

        self.depth_encoder = networks.ResnetEncoder(18, True)
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4))

        model_name = 'mono+stereo_640x192'
        model_path = os.path.join("models","monodepth2_weights", model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_encoder.eval()
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.eval()

        self.train_loss = []
        self.loss_decomp = {'seed':[], 'dCRF':[], 'proj':[]}
        self.val_loss = []
        self.test_loss = []
        self.rloss_weight = 2e-9 #2e-9
        self.rloss_scale = 0.5
        self.rloss_sig_rgb = 15
        self.rloss_sig_xy = 100
        self.ploss_weight = 0.5
        self.lr = lr
        self.width = width
        self.height = height
        self.densecrflosslayer = DenseCRFLoss(weight=self.rloss_weight, 
                                              sigma_rgb=self.rloss_sig_rgb, 
                                              sigma_xy=self.rloss_sig_xy, 
                                              scale_factor=self.rloss_scale)
        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.project_3d = Project3D(batch_size, height, width)
        self.ssim = SSIM()
        self.no_ssim = True

    def forward(self, x):
        return self.model(x) 

    def configure_optimizers(self):
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def reprojection_loss(self, seg_left, seg_right, depth_output, cam):
        disp = F.interpolate(depth_output[("disp", 0)], 
                             size=seg_left.shape[2:], 
                             mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp, 0.1, 100)
        T = cam['stereo_T']
        cam_points = self.backproject_depth(depth, cam['inv_K'])
        pix_coords = self.project_3d(cam_points, cam['K'], T)
        pred_seg_s = F.grid_sample(seg_left, pix_coords, padding_mode="border")

        reprojection_loss = self.compute_reprojection_loss(pred_seg_s, seg_right)
        return reprojection_loss


    def get_loss(self, batch):
        """Assume batch size of 2, being the stereo pair."""
        x, seeds, cam = batch
        batch_size, _, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        batch_size, _, num_classes, height, width = seeds.shape
        seeds = seeds.view(-1, num_classes, height, width)
        seg = self(x)

        x_left = x[0::2,::]
        x_right = x[1::2,::]
        seg_left = seg[0::2,::]
        seg_right = seg[1::2,::]
        features = self.depth_encoder(x_left)
        depth_output = self.depth_decoder(features)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes)
        seeds_flat = torch.argmax(seeds, dim=1)
        seed_loss = criterion(seg[0:1,::], seeds_flat[0:1,::])
        self.loss_decomp['seed'] += [seed_loss.detach()]

        if self.rloss_weight != 0:
            probs = nn.Softmax(dim=1)(seg)
            resize_img = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
            roi = torch.ones_like(seeds_flat)
            roi = resize_img(roi.unsqueeze(1).float()).squeeze(1)
            denormalized_image = denormalizeimage(x, mean=mean, std=std)
            densecrfloss = self.densecrflosslayer(denormalized_image, probs, roi).item()
            self.loss_decomp['dCRF'] += [densecrfloss.detach()]
        else:
            densecrfloss = 0
            self.loss_decomp['dCRF'] += [0]
        
        if self.ploss_weight != 0:
            p_loss = self.reprojection_loss(seg_left, seg_right, depth_output, cam)
            import IPython; IPython.embed()
            self.loss_decomp['proj'] += [p_loss.detach()]
        else:
            p_loss = 0
            self.loss_decomp['dCRF'] += [0]

        loss = seed_loss + densecrfloss + self.ploss_weight * p_loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.train_loss += [loss.detach()]
        logs = {
            'loss': loss.detach(),
            'seed': self.loss_decomp['seed'][-1],
            'dCRF': self.loss_decomp['dCRF'][-1],
            'proj': self.loss_decomp['proj'][-1]
        }
        return {
            'loss': loss,
            'log':logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        self.val_loss += [loss.detach()]
        return result

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult()
        self.test_loss += [loss.detach()]
        return result