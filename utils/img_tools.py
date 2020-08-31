import numpy as np
import torch
import matplotlib.pyplot as plt

def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def peak(seg_model, images):
  features = seg_model.encoder(images)
  depth_dict = seg_model.depth_decoder(features)
  depth_maps = {k:d_map.detach().cpu().numpy()[0] for k, d_map in depth_dict.items()}
  feat = [f.detach().cpu().numpy()[0] for f in features]
  plt.show()
  plt.imshow(np.sum(feat[-2], axis=0));
  plt.show()
  plt.imshow(np.sum(feat[-1], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('debug_1', 4)], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('debug_2', 4)], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('debug', 4)], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('debug_0', 3)], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('debug_1', 3)], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('debug_2', 3)], axis=0));
  plt.show()
  plt.imshow(np.sum(depth_maps[('disp', 3)], axis=0));
  plt.show()
  plt.imshow(depth_maps[('disp', 0)][0]);
  plt.show()