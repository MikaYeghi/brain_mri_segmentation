import torch
import os, glob
from dataset import MRIDataset
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, FocalLoss, TverskyLoss
from torch import optim
from tqdm import tqdm
import config

from model import init_backbone, MRIModel
from utils import make_train_step

from matplotlib import pyplot as plt
import pdb

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
batch_size = config.BATCH_SIZE
lr = config.LR
n_epochs = config.N_EPOCHS
save_path = config.SAVE_PATH
val_freq = config.VAL_FREQ
model_name = config.MODEL_NAME
load_model = config.LOAD_MODEL
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Load the data set"""
train_img_path = os.path.join(ROOT_PATH, "data/train/image/")
train_mask_path = os.path.join(ROOT_PATH, "data/train/mask/")
val_img_path = os.path.join(ROOT_PATH, "data/val/image/")
val_mask_path = os.path.join(ROOT_PATH, "data/val/mask/")

# Create the Dataset classes for the train and validation data sets
train_data = MRIDataset(train_img_path, train_mask_path, device)
val_data = MRIDataset(val_img_path, val_mask_path, device)
print(f"Loaded a training data set with {len(train_data)} images.")
print(f"Loaded a validation data set with {len(train_data)} images.")

"""Create the dataloaders"""
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

"""Initialize the model"""
backbone = init_backbone(in_channels=3, classes=1, device=device)
model = MRIModel(backbone=backbone, device=device, save_path=save_path)
if load_model:
    model_path = os.path.join(save_path, model_name)
    model.load(model_path)

"""Define the loss function and the optimizer"""
loss_fn = FocalLoss(
    mode='binary',
    alpha=0.85,
)

optimizer = optim.SGD(
    model.parameters(),
    lr=lr
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 
    gamma=0.1
)

"""Start training"""
train_step = make_train_step(model, loss_fn, optimizer)
losses = list()
print("-" * 80)
for epoch in range(n_epochs):

    for images_batch, masks_batch in tqdm(train_loader):
        images_batch = model.preprocess_batch(images_batch)
        masks_batch = model.preprocess_batch(masks_batch)
        images_batch = images_batch.permute(0, 3, 1, 2)
        loss = train_step(images_batch, masks_batch)
        losses.append(loss)

        print(f"Epoch: #{epoch + 1}. Loss: {round(loss, 5)}. LR: {optimizer.state_dict()['param_groups'][0]['lr']}.")
        print("-" * 80)

    if epoch % val_freq == 0:
        if epoch != 0:
            scheduler.step()
        """Add validation code"""
        print("Saving the model...")
        model.save()

    print("-" * 80)