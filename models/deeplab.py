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

from collections import OrderedDict
from utils import denormalizeimage

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class DeeplabModel(pl.LightningModule):

    def __init__(self, lr=7e-3):
        super().__init__()
        self.num_classes = 9
        self.model = DeepLab(num_classes=self.num_classes)

        self.train_loss = []
        self.loss_decomp = {'seed':[], 'dCRF':[]}
        self.val_loss = []
        self.test_loss = []
        self.rloss_weight = 2e-9 #2e-9
        self.rloss_scale = 0.5
        self.rloss_sig_rgb = 15
        self.rloss_sig_xy = 100
        self.lr = lr
        self.densecrflosslayer = DenseCRFLoss(weight=self.rloss_weight, 
                                              sigma_rgb=self.rloss_sig_rgb, 
                                              sigma_xy=self.rloss_sig_xy, 
                                              scale_factor=self.rloss_scale)

    def forward(self, x):
        return self.model(x) 

    def configure_optimizers(self):
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def get_loss(self, batch):
        x, seeds = batch
        seg = self(x)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes)
        seeds = torch.argmax(seeds, dim=1)
        seed_loss = criterion(seg, seeds)
        self.loss_decomp['seed'] += [seed_loss.detach()]
        if self.rloss_weight == 0:
          self.loss_decomp['dCRF'] += [0]
          return seed_loss

        probs = nn.Softmax(dim=1)(seg)
        resize_img = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
        # probs = resize_img(probs)
        roi = resize_img(seeds.unsqueeze(1).float()).squeeze(1)
        denormalized_image = denormalizeimage(x, mean=mean, std=std)
        densecrfloss = self.densecrflosslayer(denormalized_image, probs, roi)
        self.loss_decomp['dCRF'] += [densecrfloss.detach()]
        loss = seed_loss + densecrfloss.item()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.train_loss += [loss.detach()]
        logs = {
            'loss': loss.detach(),
            'seed': self.loss_decomp['seed'][-1],
            'dCRF': self.loss_decomp['dCRF'][-1]
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