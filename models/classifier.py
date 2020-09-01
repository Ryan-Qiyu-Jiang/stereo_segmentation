import numpy as np
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import pytorch_lightning as pl
import sys, os

import stero_segmentation.monodepth2.networks as networks
from stero_segmentation.monodepth2.layers import *
from collections import OrderedDict

class ClassifierModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 9
        self.cam_thresh = 0.9
        self.encoder_cls = networks.ResnetEncoder(18, True)

        self.encoder_out_channels = self.encoder_cls.num_ch_enc[-1]
        self.classifer_conv1 = nn.Conv2d(self.encoder_out_channels, 1024, 3, padding=1)
        self.classifer_drop1 = nn.Dropout2d(p=0.5)
        self.classifer_conv2 = nn.Conv2d(1024, self.num_classes, 1, bias=False)
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.lr = 1e-3

    def get_seeds(self, x):
      _, cam = self(x)
      batch_size, num_class, h, w = cam.shape
      seeds = torch.zeros_like(cam)
      one = torch.ones(1)
      zero = torch.zeros(batch_size, 1, h, w)
      if x.is_cuda:
        one = one.cuda()
        zero = zero.cuda()

      for i in range(batch_size):
        seeds[i] = torch.where(cam[i] > 0.9 * torch.max(cam[i]), one, -one)
      seeds = torch.cat((seeds, zero), 1)
      seeds[:,0,::] = torch.where(torch.all(cam[:] < 0, dim=1), one, -one)
      seeds = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)(seeds.type(torch.DoubleTensor))
      seeds = seeds.type(torch.LongTensor).cuda()
      return seeds

    def forward(self, x):
        features_cls = self.encoder_cls(x)
        feat = self.classifer_drop1(F.relu(self.classifer_conv1(features_cls[-1])))
        cam = self.classifer_conv2(feat)
        score = F.avg_pool2d(cam, kernel_size=(cam.size(2), cam.size(3)), padding=0)
        score = score.view(score.size(0), -1)
        return (score, cam)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def get_loss(self, batch):
        x, label = batch
        score, _ = self(x)
        cls_loss = F.multilabel_soft_margin_loss(score, label)
        return cls_loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.train_loss += [loss]
        logs = { 'training_loss': loss.detach() }
        return {
            'loss': loss,
            'log':logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        self.val_loss += [loss]
        return result

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        self.test_loss += [loss]
        return result