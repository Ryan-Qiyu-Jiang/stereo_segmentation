import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.patches as patches
from skimage.transform import resize
import sys, os
import pandas as pd
import numpy as np

CLASS_NAMES = ['DontCare','Car', 'Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram','Misc']
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.max_objects = 50

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)

        transform = transforms.Compose([transforms.Resize((192, 640), interpolation=2),
                                        transforms.RandomHorizontalFlip(p=0.5), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        input_img = transform(img)
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
          names = pd.read_csv(label_path, header=None, sep="\s+", usecols=[0], names=['class'])['class'].values
          indexs = np.array([CLASS_NAMES.index(n) for n in names])
          labels = np.zeros(len(CLASS_NAMES))
          labels[indexs] = 1
          labels[0] = 0
          # IPython.embed()
          labels = torch.LongTensor(labels)
        
        return input_img, labels

    def __len__(self):
        return len(self.img_files)

# /content/kitti/images/train/001187.png

class SeedsDataset(torch.utils.data.Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.max_objects = 50

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)
        transform = transforms.Compose([transforms.Resize((192, 640), interpolation=2),
                                        # transforms.RandomHorizontalFlip(p=0.5), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        input_img = transform(img)

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        seeds = None
        index = None
        if os.path.exists(label_path):
          labels_df = pd.read_csv(label_path, header=None, sep="\s+")
          labels_df[0] = np.array([CLASS_NAMES.index(n) for n in labels_df[0]])
          seeds = torch.zeros(len(CLASS_NAMES)+1, 192, 640)
          w, h = (640, 192)
          w1, h1 = img.size
          bbox = labels_df.iloc[:, 4:8].values
          index = np.zeros((bbox.shape[0], 3))
          index[:, 0] = labels_df[0].values
          index[:, 2] = (bbox[:,0]+bbox[:,2])/2/w1*w
          index[:, 1] = (bbox[:,1]+bbox[:,3])/2/h1*h
          index = np.round(index).astype(int)
          for img_class, y, x in index:
            seeds[img_class,y-5:y+5,x-5:x+5] = 1
          seeds[len(CLASS_NAMES)] = 0.5
        
        return input_img, seeds

    def __len__(self):
        return len(self.img_files)

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, image, seeds, length):
        self.image = image
        self.seeds = seeds
        self.length = length

    def __getitem__(self, index):
        return self.image, self.seeds

    def __len__(self):
        return self.length

class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.max_objects = 50

    def __getitem__(self, data_index):
        
        file_index = int(data_index/2)
        use_pair = data_index % 2 == 0
        img_path = self.img_files[file_index % len(self.img_files)].rstrip()
        if use_pair:
            img_path = img_path.replace('images', 'right_images')
        img = Image.open(img_path)
        transform = transforms.Compose([transforms.Resize((192, 640), interpolation=2),
                                        # transforms.RandomHorizontalFlip(p=0.5), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        input_img = transform(img)

        if not use_pair:
            label_path = self.label_files[file_index % len(self.img_files)].rstrip()
            seeds = None
            index = None
            if os.path.exists(label_path):
                labels_df = pd.read_csv(label_path, header=None, sep="\s+")
                labels_df[0] = np.array([CLASS_NAMES.index(n) for n in labels_df[0]])
                seeds = torch.zeros(len(CLASS_NAMES)+1, 192, 640)
                w, h = (640, 192)
                w1, h1 = img.size
                bbox = labels_df.iloc[:, 4:8].values
                index = np.zeros((bbox.shape[0], 3))
                index[:, 0] = labels_df[0].values
                index[:, 2] = (bbox[:,0]+bbox[:,2])/2/w1*w
                index[:, 1] = (bbox[:,1]+bbox[:,3])/2/h1*h
                index = np.round(index).astype(int)
                for img_class, y, x in index:
                    seeds[img_class,y-5:y+5,x-5:x+5] = 1
                seeds[len(CLASS_NAMES)] = 0.5
        else:
            seeds = torch.zeros(len(CLASS_NAMES)+1, 192, 640)
            seeds[len(CLASS_NAMES)] = 0.5
        
        return input_img, seeds

    def __len__(self):
        return 2*len(self.img_files)

# train_path = '/content/kitti/train.txt'
# train_dataset = SingleDataset(train_path)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
# images, labels = iter(train_loader).next()
