import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.densecrflosslayer = DenseCRFLoss(weight=1, 
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

        reprojection_loss = self.compute_reprojection_loss(pred_seg_s, seg_right).mean()
        return reprojection_loss

    def get_rloss(self, seg, x, depth_output):
        probs = nn.Softmax(dim=1)(seg)
        resize_img = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
        batch_size, num_classes, h, w = seg.shape
        # roi = torch.ones(batch_size, h, w)
        roi = seg[:,0,::].detach().cpu()
        max_roi = torch.max(roi)
        min_roi = torch.min(roi)
        roi = 1-(roi-min_roi)/(max_roi-min_roi)
        roi = resize_img(roi.unsqueeze(1).float()).squeeze(1)
        disp = F.interpolate(depth_output[("disp", 0)], 
                             size=x.shape[2:], 
                             mode="bilinear", align_corners=False)
        min_disp = torch.min(disp)
        max_disp = torch.max(disp)
        disp = (disp-min_disp)/(max_disp-min_disp)*255.0
        disp_img = torch.cat([disp, disp, disp], dim=1).detach()
        # import IPython; IPython.embed()
        densecrfloss = self.densecrflosslayer(disp_img.cpu(), probs, roi)
        return self.rloss_weight*densecrfloss

    def get_loss(self, batch):
        """Assume batch size of 2, being the stereo pair."""
        x, seeds, cam = batch
        batch_size, _, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        batch_size, _, num_classes, height, width = seeds.shape
        seeds = seeds.view(-1, num_classes, height, width)
        seg = self(x)

        x_left = x[0::2,::]
        seg_left = seg[0::2,::]
        seg_right = seg[1::2,::]
        seeds_left = seeds[0::2,::]

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes)
        seeds_flat = torch.argmax(seeds_left, dim=1)
        seed_loss = criterion(seg_left, seeds_flat)
        self.loss_decomp['seed'] += [seed_loss.detach()]

        features = None
        depth_output = None
        if self.rloss_weight != 0 or self.ploss_weight != 0 :
            features = self.depth_encoder(x_left)
            depth_output = self.depth_decoder(features)

        if self.rloss_weight != 0:
            densecrfloss = self.get_rloss(seg_left, x_left, depth_output)
            if seed_loss.is_cuda:
                densecrfloss = densecrfloss.cuda()
            self.loss_decomp['dCRF'] += [densecrfloss.detach()]
            # import IPython; IPython.embed()
        else:
            densecrfloss = 0
            self.loss_decomp['dCRF'] += [0]
        
        if self.ploss_weight != 0:
            p_loss = self.reprojection_loss(seg_left, seg_right, depth_output, cam)
            self.loss_decomp['proj'] += [p_loss.detach()]
        else:
            p_loss = 0
            self.loss_decomp['proj'] += [0]
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
        self.val_loss += [loss.detach()]
        logs = {
            'val_loss': loss.detach(),
            'val_seed': self.loss_decomp['seed'][-1],
            'val_dCRF': self.loss_decomp['dCRF'][-1],
            'val_proj': self.loss_decomp['proj'][-1]
        }
        return {
            'loss': loss,
            'log':logs
        }

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult()
        self.test_loss += [loss.detach()]
        return result


class ProjectionBottleneckModel(StereoProjectionModel):

    def __init__(self, lr=7e-3, batch_size=1, width=640, height=192):
        super().__init__(lr, batch_size, width, height)

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

    def get_segment_disp(self, seg, disp, threshold=0.1, terrible_disp=100):
        with torch.no_grad():
            probs = nn.Softmax(dim=1)(seg)
            batch_size, num_classes, height, width = seg.shape
            seg_disp = torch.ones_like(seg)*terrible_disp
            disp_max_pool = nn.MaxPool2d(100, stride=100)(disp)
            disp_window_max = F.interpolate(disp_max_pool, 
                                size=seg.shape[2:], 
                                mode="bilinear", align_corners=False)
            disp_max_stacked = torch.cat([disp_window_max for _ in range(self.num_classes)], dim=1)
            seg_disp[probs > threshold] = disp_max_stacked[probs > threshold]
        return seg_disp
        

    def reprojection_loss(self, x_left, x_right, seg_left, depth_output, cam):
        disp = F.interpolate(depth_output[("disp", 0)], 
                             size=seg_left.shape[2:], 
                             mode="bilinear", align_corners=False)
        seg_disp = self.get_segment_disp(seg_left, disp)
        reprojection_loss = torch.zeros(1)
        if seg_left.is_cuda:
            seg_disp = seg_disp.cuda()
            reprojection_loss = reprojection_loss.cuda()

        _, seg_depths = disp_to_depth(seg_disp, 0.1, 100)
        T = cam['stereo_T']
        for class_index in range(1, self.num_classes):
            depth = seg_depths[:,class_index, ::]
            cam_points = self.backproject_depth(depth, cam['inv_K'])
            pix_coords = self.project_3d(cam_points, cam['K'], T)
            pred_x_right = F.grid_sample(x_left, pix_coords, padding_mode="border")
            reprojection_loss += self.compute_reprojection_loss(pred_x_right, x_right).mean()
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
        seeds_left = seeds[0::2,::]

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes)
        seeds_flat = torch.argmax(seeds_left, dim=1)
        seed_loss = criterion(seg_left, seeds_flat)
        self.loss_decomp['seed'] += [seed_loss.detach()]

        features = None
        depth_output = None
        if self.ploss_weight != 0 :
            features = self.depth_encoder(x_left)
            depth_output = self.depth_decoder(features)
        
        if self.ploss_weight != 0:
            p_loss = self.reprojection_loss(x_left, x_right, seg_left, depth_output, cam)
            self.loss_decomp['proj'] += [p_loss.detach()]
        else:
            p_loss = 0
            self.loss_decomp['proj'] += [0]
        loss = seed_loss + self.ploss_weight * p_loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.train_loss += [loss.detach()]
        logs = {
            'loss': loss.detach(),
            'seed': self.loss_decomp['seed'][-1],
            'proj': self.loss_decomp['proj'][-1]
        }
        return {
            'loss': loss,
            'log':logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.val_loss += [loss.detach()]
        logs = {
            'val_loss': loss.detach(),
            'val_seed': self.loss_decomp['seed'][-1],
            'val_proj': self.loss_decomp['proj'][-1]
        }
        return {
            'loss': loss,
            'log':logs
        }
