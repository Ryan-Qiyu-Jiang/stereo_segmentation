import numpy as np
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import pytorch_lightning as pl
import sys
sys.path.append(os.path.abspath("../rloss/pytorch/pytorch-deeplab_v3_plus"))
from DenseCRFLoss import DenseCRFLoss
sys.path.append(os.path.abspath("../monodepth2"))
import networks

import monodepth2.networks as networks
from stero_segmentation.monodepth2.layers import *
from collections import OrderedDict

class JointModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.num_classes = 9
        self.cam_thresh = 0.9
        self.seg_ratio = 0.2
        self.encoder = networks.ResnetEncoder(18, False)
        self.encoder_cls = networks.ResnetEncoder(18, True)

        model_name = 'mono+stereo_640x192'
        model_path = os.path.join("models", model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        # self.encoder.eval();
        
        self.decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(5),
            num_output_channels=self.num_classes)
        
        self.depth_decoder = DebugDepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        self.depth_decoder.load_state_dict(loaded_dict)
        # self.decoder.eval();

        self.encoder_out_channels = self.encoder.num_ch_enc[-1]
        self.classifer_conv1 = nn.Conv2d(self.encoder_out_channels, 1024, 3, padding=1)
        self.classifer_drop1 = nn.Dropout2d(p=0.5)
        self.classifer_conv2 = nn.Conv2d(1024, self.num_classes, 1, bias=False)
        # self.seg_conv1 = nn.Conv2d(self.encoder_out_channels, self.num_classes, 1)
        self.train_loss = []
        self.loss_decomp = {'cls':[], 'seed':[], 'dCRF':[]}
        self.val_loss = []
        self.test_loss = []
        self.rloss_weight = 2e-9 #2e-9
        self.rloss_scale = 0.5
        self.rloss_sig_rgb = 15
        self.rloss_sig_xy = 100
        self.lr = 1e-3
        self.densecrflosslayer = DenseCRFLoss(weight=self.rloss_weight, 
                                              sigma_rgb=self.rloss_sig_rgb, 
                                              sigma_xy=self.rloss_sig_xy, 
                                              scale_factor=self.rloss_scale)

    def forward_full(self, x):
        # import IPython ; IPython.embed()
        features_cls = self.encoder_cls(x)
        
        feat = self.classifer_drop1(F.relu(self.classifer_conv1(features_cls[-1])))
        cam = self.classifer_conv2(feat)
        score = F.avg_pool2d(cam, kernel_size=(cam.size(2), cam.size(3)), padding=0)
        score = score.view(score.size(0), -1)

        if self.seg_ratio == 0:
          return (score, None, cam, None, None)

        features = self.encoder(x)
        seg_dict = self.decoder(features)
        seg = seg_dict[('disp', 0)]

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
        seeds = nn.Upsample(size=seg.shape[2:], mode='bilinear', align_corners=True)(seeds.type(torch.DoubleTensor))
        seeds = seeds.type(torch.LongTensor).cuda()

        return (score, seg, cam, seeds, features)

    def forward(self, x):
        score, seg, cam, seeds, _ = self.forward_full(x)

        return (score, seg, cam, seeds)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def get_loss(self, batch):
        x, label = batch
        score, seg, cam, seeds = self(x)
        cls_loss = F.multilabel_soft_margin_loss(score, label)
        if self.seg_ratio == 0:
          return cls_loss

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.num_classes)
        seeds = torch.argmax(seeds, dim=1)
        seg_padded = torch.cat((seg, seg[:,:1,::]), dim=1)
        seed_loss = criterion(seg_padded, seeds)
        
        probs = nn.Softmax(dim=1)(seg)
        resize_img = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
        probs = resize_img(probs)
        roi = resize_img(seeds.unsqueeze(1).float()).squeeze(1)
        denormalized_image = denormalizeimage(x, mean=mean, std=std)
        densecrfloss = self.densecrflosslayer(denormalized_image,probs,roi)
        self.loss_decomp['cls'] += [cls_loss.detach()]
        self.loss_decomp['seed'] += [seed_loss.detach()]
        self.loss_decomp['dCRF'] += [densecrfloss.detach()]
        seed_loss += densecrfloss.item()
        loss = cls_loss + self.seg_ratio * seed_loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.train_loss += [loss.detach()]
        logs = {
            'loss': loss.detach(),
            'cls': self.loss_decomp['cls'][-1].detach(),
            'seed': self.loss_decomp['seed'][-1].detach(),
            'dCRF': self.loss_decomp['dCRF'][-1].detach()
        }
        return {
            'loss': loss,
            'log':logs
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        # result.log('val_loss', loss)
        self.val_loss += [loss.detach()]
        return result

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        result = pl.EvalResult()
        # result.log('test_loss', loss)
        self.test_loss += [loss.detach()]
        return result