from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import random_split
from torchvision import transforms, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime, os
from pytorch_lightning import loggers as pl_loggers

from datasets import SingleDataset, SeedsDataset, ListDataset
from models import SegModel, SegSeedModel, JointModel, DebugDepthDecoder, ClassifierModel
from utils import denormalizeimage, show

device = torch.device("cuda")


train_path = 'data/kitti/train.txt'
train_dataset = SeedsDataset(train_path)
n = len(train_dataset)
train, val = random_split(train_dataset, [int(n*0.7), n-int(n*0.7)])
train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val, batch_size=1, num_workers=4)

images, labels = iter(train_loader).next()
single_dataset = SingleDataset(images, labels, n)
single_loader = DataLoader(single_dataset, batch_size=1, shuffle=True, num_workers=4)

# joint_model = JointModel()
# cls_model = ClassifierModel()
# cls_model = ClassifierModel.load_from_checkpoint(checkpoint_path='cls.ckpt').to('cuda')
# seg_model = SegModel(cls_model)
overfit_seg_model = SegSeedModel()
overfit_seg_model.rloss_weight = 0
overfit_seg_model.lr = 7e-3

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_parameters(seg_model))

tb_logger = pl_loggers.TensorBoardLogger('/logs/')

trainer = pl.Trainer(gpus=1, max_epochs=10, 
                     progress_bar_refresh_rate=100, 
                     logger=tb_logger)
trainer.fit(overfit_seg_model, single_loader, val_loader)