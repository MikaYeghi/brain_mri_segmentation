import torch
import os, glob
from dataset import MRIDataset
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import JaccardLoss
from torch import optim
from tqdm import tqdm
import config

from model import init_backbone, MRIModel
from utils import make_train_step

from matplotlib import pyplot as plt

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
batch_size = config.BATCH_SIZE
lr = config.LR
n_epochs = config.N_EPOCHS
save_path = config.SAVE_PATH
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Load the data set"""
train_img_path = os.path.join(ROOT_PATH, "data/train/image/")
train_mask_path = os.path.join(ROOT_PATH, "data/train/mask/")
val_img_path = os.path.join(ROOT_PATH, "data/val/image/")
val_mask_path = os.path.join(ROOT_PATH, "data/val/mask/")

# Create the Dataset classes for the train and validation data sets
train_data = MRIDataset(train_img_path, train_mask_path, device)
val_data = MRIDataset(val_img_path, val_mask_path, device)

"""Create the dataloaders"""
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

"""Initialize the model"""
backbone = init_backbone(in_channels=3, classes=1, device=device)
model = MRIModel(backbone=backbone, device=device, save_path=save_path)

"""Define the loss function and the optimizer"""
loss_fn = JaccardLoss(
    mode='binary'
)
optimizer = optim.SGD(
    model.parameters(),
    lr=lr
)

"""Start training"""
train_step = make_train_step(model, loss_fn, optimizer)
losses = list()
for epoch in tqdm(range(n_epochs)):
    print(f"Epoch #{epoch + 1}")

    for images_batch, masks_batch in tqdm(train_loader):
        images_batch = model.preprocess_batch(images_batch)
        images_batch = images_batch.permute(0, 3, 1, 2)
        loss = train_step(images_batch, masks_batch)
        losses.append(loss)
        print(f"Loss: {round(loss, 5)}")
    
    print("Saving the model...")
    model.save()
    print("-" * 80)