import torch
import os
from dataset import MRIDataset
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import FocalLoss
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
gamma = config.SCHEDULER_GAMMA
n_epochs = config.N_EPOCHS
save_path = config.SAVE_PATH
val_freq = config.VAL_FREQ
model_name = config.MODEL_NAME
load_model = config.LOAD_MODEL
scheduler_step = config.SCHEDULER_STEP
encoder = config.ENCODER
FREEZE_ENCODER = config.FREEZE_ENCODER
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
print(f"Loaded a validation data set with {len(val_data)} images.")

"""Create the dataloaders"""
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

"""Initialize the model"""
backbone = init_backbone(in_channels=3, classes=1, device=device, encoder=encoder)
model = MRIModel(backbone=backbone, device=device, save_path=save_path)
if load_model:
    model_path = os.path.join(save_path, model_name)
    model.load(model_path)
if FREEZE_ENCODER:
    model.freeze_encoder()

"""Define the loss function and the optimizer"""
loss_fn = FocalLoss(
    mode='binary',
    alpha=0.564
)

optimizer = optim.Adam(
    model.parameters(),
    lr=lr
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 
    gamma=gamma
)

"""Training"""
train_step = make_train_step(model, loss_fn, optimizer)
train_losses = list()
val_losses = list()
print("-" * 80)

for epoch in range(n_epochs):

    for images_batch, masks_batch in tqdm(train_loader):
        images_batch = model.preprocess_batch(images_batch)
        masks_batch = model.preprocess_batch(masks_batch)
        images_batch = images_batch.permute(0, 3, 1, 2)
        
        loss = train_step(images_batch, masks_batch)
        train_losses.append(loss)

        print(f"Epoch: #{epoch + 1}. Loss: {round(loss, 5)}. LR: {optimizer.state_dict()['param_groups'][0]['lr']}.")
        print("-" * 80)

    if (epoch + 1) % val_freq == 0:
        print("Runnning validation...")
        with torch.no_grad():
            for images_batch, masks_batch in tqdm(val_loader):
                images_batch = model.preprocess_batch(images_batch)
                masks_batch = model.preprocess_batch(masks_batch)
                images_batch = images_batch.permute(0, 3, 1, 2)

                model.eval()
                preds = model(images_batch)
                val_loss = loss_fn(masks_batch, preds)
                val_losses.append(val_loss.item())
                print(f"Validation loss: {round(val_loss.item(), 5)}.")

        print("Saving the model...")
        model.save()
    
    if (epoch + 1) % scheduler_step == 0 and epoch != 0:
        scheduler.step()

    print("-" * 80)