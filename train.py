import torch
import os, glob
from dataset import MRIDataset

import config

"""Load the data set"""
train_img_path = os.path.join(config.ROOT_PATH, "data/train/image/")
train_mask_path = os.path.join(config.ROOT_PATH, "data/train/mask/")

train_data = MRIDataset(train_img_path, train_mask_path)
image, mask = train_data[1]