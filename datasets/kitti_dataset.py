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
import copy

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
    def __init__(self, image, seeds, cam, length):
        self.image = image
        self.seeds = seeds
        self.cam = cam
        self.length = length

    def __getitem__(self, index):
        return self.image, self.seeds, self.cam

    def __len__(self):
        return self.length


class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, list_path, width=640, height=192):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.max_objects = 50
        self.width = width
        self.height = height
        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height

        inv_K = np.linalg.pinv(K)

        self.cam = {}
        self.cam["K"] = torch.from_numpy(K)
        self.cam["inv_K"] = torch.from_numpy(inv_K)
        
    def get_stereo_T(self, side='l', do_flip=False):
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.1

        return torch.from_numpy(stereo_T)

    def get_bg_seeds(self, bbox, w1, h1, num_seeds=1):
        potential_points = np.random.rand(10, 2) # 10 random points x, y
        valid_points_mask = np.ones(10, dtype=bool)
        for i in range(len(bbox)):
            not_in_box_mask = ((potential_points[:,0] > bbox[i,2]/w1) | 
                                    (potential_points[:,0] < bbox[i,0]/w1) |
                                    (potential_points[:,1] > bbox[i,3]/h1) |
                                    (potential_points[:,1] < bbox[i,1]/h1))
            valid_points_mask &= not_in_box_mask
        return potential_points[valid_points_mask][:num_seeds]

    def __getitem__(self, file_index):
        np.random.seed(file_index)
        left_img_path = self.img_files[file_index % len(self.img_files)].rstrip()
        right_img_path = left_img_path.replace('images', 'right_images')
        img_left = Image.open(left_img_path)
        img_right = Image.open(right_img_path)

        transform = transforms.Compose([transforms.Resize((self.height, self.width), interpolation=2),
                                        # transforms.RandomHorizontalFlip(p=0.5), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        input_img_left = transform(img_left)
        input_img_right = transform(img_right)

        label_path = self.label_files[file_index % len(self.img_files)].rstrip()
        seeds = None
        index = None
        if os.path.exists(label_path):
            labels_df = pd.read_csv(label_path, header=None, sep="\s+")
            labels_df[0] = np.array([CLASS_NAMES.index(n) for n in labels_df[0]])
            seeds = torch.zeros(len(CLASS_NAMES)+1, 192, 640)
            w, h = (640, 192)
            w1, h1 = img_left.size
            bbox = labels_df.iloc[:, 4:8].values
            index = np.zeros((bbox.shape[0], 3))
            index[:, 0] = labels_df[0].values
            index[:, 2] = (bbox[:,0]+bbox[:,2])/2/w1*w
            index[:, 1] = (bbox[:,1]+bbox[:,3])/2/h1*h
            index = np.round(index).astype(int)
            for img_class, y, x in index:
                if img_class != 0:
                    seeds[img_class,y-5:y+5,x-5:x+5] = 1

            index = np.zeros((bbox.shape[0], 3))
            index[:, 0] = labels_df[0].values
            index[:, 2] = (bbox[:,0]/4+bbox[:,2]*3/4)/w1*w
            index[:, 1] = (bbox[:,1]+bbox[:,3])/2/h1*h
            index = np.round(index).astype(int)
            for img_class, y, x in index:
                if img_class != 0:
                    seeds[img_class,y-3:y+3,x-3:x+3] = 1

            index = np.zeros((bbox.shape[0], 3))
            index[:, 0] = labels_df[0].values
            index[:, 2] = (bbox[:,0]*3/4+bbox[:,2]/4)/w1*w
            index[:, 1] = (bbox[:,1]+bbox[:,3])/2/h1*h
            index = np.round(index).astype(int)
            for img_class, y, x in index:
                if img_class != 0:
                    seeds[img_class,y-3:y+3,x-3:x+3] = 1

            seeds[len(CLASS_NAMES)] = 0.5
            bg_seeds = self.get_bg_seeds(bbox, w1, h1, num_seeds=5)
            for x_frac, y_frac in bg_seeds:
                x = int(x_frac*w)
                y = int(y_frac*h)
                seeds[0, y-5:y+5, x-5:x+5] = 1
        
        seeds_empty = torch.zeros(len(CLASS_NAMES)+1, 192, 640)
        seeds_empty[len(CLASS_NAMES)] = 0.5

        seeds_left = torch.unsqueeze(seeds, dim=0)
        seeds_right = torch.unsqueeze(seeds_empty, dim=0)
        seeds_pair = torch.cat([seeds_left, seeds_right], dim=0)

        tensor_left = torch.unsqueeze(input_img_left, dim=0)
        tensor_right = torch.unsqueeze(input_img_right, dim=0)
        img_pair = torch.cat([tensor_left, tensor_right], dim=0)

        cam = copy.deepcopy(self.cam)
        cam['stereo_T'] = self.get_stereo_T(side='l', do_flip=False)

        return img_pair, seeds_pair, cam

    def __len__(self):
        return 2*len(self.img_files)

# train_path = '/content/kitti/train.txt'
# train_dataset = SingleDataset(train_path)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
# images, labels = iter(train_loader).next()
