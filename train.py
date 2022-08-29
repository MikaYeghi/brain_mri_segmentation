import torch
import os, glob
from dataset import MRIDataset
from torch.utils.data import DataLoader

import config

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Load the data set"""
train_img_path = os.path.join(ROOT_PATH, "data/train/image/")
train_mask_path = os.path.join(ROOT_PATH, "data/train/mask/")
val_img_path = os.path.join(ROOT_PATH, "data/val/image/")
val_mask_path = os.path.join(ROOT_PATH, "data/val/mask/")

# Create the Dataset classes for the train and validation data sets
train_data = MRIDataset(train_img_path, train_mask_path)
val_data = MRIDataset(val_img_path, val_mask_path)

"""Create the dataloaders"""
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

"""Initialize the model"""
pass