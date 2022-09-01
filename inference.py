import os
import torch
from dataset import MRIDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import init_backbone, MRIModel
from utils import evaluate, save_predictions

import config

import pdb

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
batch_size = config.BATCH_SIZE
save_path = config.SAVE_PATH
model_name = config.MODEL_NAME
save_preds = config.SAVE_PREDS
preds_path = config.PREDS_PATH
rounded_save = config.ROUNDED_SAVE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
k = 0

"""Load the data set on which inference is run"""
test_img_path = os.path.join(ROOT_PATH, "data/test/image/")
test_mask_path = os.path.join(ROOT_PATH, "data/test/mask/")

# Create the Dataset classes for the train and validation data sets
test_data = MRIDataset(test_img_path, test_mask_path, device)
print(f"Loaded a test data set with {len(test_data)} images.")

"""Create the dataloader"""
test_loader = DataLoader(test_data, batch_size=batch_size)

"""Load the model"""
backbone = init_backbone(in_channels=3, classes=1, device=device)
model = MRIModel(backbone=backbone, device=device, save_path=save_path)
saved_model_path = os.path.join(save_path, model_name)
model.load(saved_model_path)

"""Run inference"""
for images_batch, masks_batch in tqdm(test_loader):
    images_batch = model.preprocess_batch(images_batch)
    masks_batch = model.preprocess_batch(masks_batch)
    images_batch = images_batch.permute(0, 3, 1, 2)

    preds = evaluate(model, images_batch)

    if save_preds:
        k = save_predictions(preds, preds_path, rounded_save, k)