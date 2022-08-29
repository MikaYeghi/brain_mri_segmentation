import os, glob
from pprint import pprint
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import pdb

import config

"""Load the data set"""
data_path = os.path.join(config.ROOT_PATH, "raw_data/kaggle_3m/")
cases = glob.glob(os.path.join(data_path, "TCGA*"))
data = list()
for case in cases:
    case_files = os.listdir(case)       # get the list of files in the current case
    masks = glob.glob(case + "/*mask.tif")
    images = [mask.split('_mask')[0] + mask.split('_mask')[1] for mask in masks]
    
    assert len(masks) == len(images)
    for i in range(len(masks)):
        data.append((images[i], masks[i]))

data = np.array(data)

"""Split the data"""
print("All paths have been collected. Ready to continue.")
print(f"Total number of images: {len(data)}.\n")

print("Splitting the data into trainval and test sets... Ratio 9:1.")
trainval_data, test_data = train_test_split(data, test_size=0.1)
print(f"Train and validation data sets size: {trainval_data.shape}.")
print(f"Test set size: {test_data.shape}.\n")

print("Splitting the data into train and validation sets... Ratio 8:2.")
train_data, val_data = train_test_split(trainval_data, test_size=0.2)
print(f"Train data size: {train_data.shape}.")
print(f"Validation data size: {val_data.shape}.\n")

"""Save the files in relevant directories"""
train_folder = os.path.join(config.ROOT_PATH, "data/train")
val_folder = os.path.join(config.ROOT_PATH, "data/val")
test_folder = os.path.join(config.ROOT_PATH, "data/test")
# Save the test data
print("Copying the test data...")
for test_data_ in tqdm(test_data):
    image = test_data_[0]
    mask = test_data_[1]
    basename = os.path.basename(image)
    shutil.copyfile(image, os.path.join(os.path.join(test_folder, "image"), basename))
    shutil.copyfile(mask, os.path.join(os.path.join(test_folder, "mask"), basename))

print("Copying the validation data...")
for val_data_ in tqdm(val_data):
    image = val_data_[0]
    mask = val_data_[1]
    basename = os.path.basename(image)
    shutil.copyfile(image, os.path.join(os.path.join(val_folder, "image"), basename))
    shutil.copyfile(mask, os.path.join(os.path.join(val_folder, "mask"), basename))

print("Copying the train data...")
for train_data_ in tqdm(train_data):
    image = train_data_[0]
    mask = train_data_[1]
    basename = os.path.basename(image)
    shutil.copyfile(image, os.path.join(os.path.join(train_folder, "image"), basename))
    shutil.copyfile(mask, os.path.join(os.path.join(train_folder, "mask"), basename))

print("Data has been successfully shuffled!")