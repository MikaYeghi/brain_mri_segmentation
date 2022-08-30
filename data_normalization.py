import torch
import os
from dataset import MRIDataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
torch.manual_seed(42)

"""CONFIG"""
ROOT_PATH = config.ROOT_PATH
batch_size = config.BATCH_SIZE
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Load the data set"""
train_img_path = os.path.join(ROOT_PATH, "data/train/image/")
train_mask_path = os.path.join(ROOT_PATH, "data/train/mask/")
val_img_path = os.path.join(ROOT_PATH, "data/val/image/")
val_mask_path = os.path.join(ROOT_PATH, "data/val/mask/")

# Create the Dataset classes for the train and validation data sets
train_data = MRIDataset(train_img_path, train_mask_path, device)
val_data = MRIDataset(val_img_path, val_mask_path, device)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

mean = torch.tensor([0., 0., 0.], device=device)
std = torch.tensor([0., 0., 0.], device=device)
for images, _ in tqdm(train_loader):
    mean[0] += torch.sum(images[:, :, :, 0])
    mean[1] += torch.sum(images[:, :, :, 1])
    mean[2] += torch.sum(images[:, :, :, 2])
    std[0] += torch.sum(torch.square(images)[:, :, :, 0])
    std[1] += torch.sum(torch.square(images)[:, :, :, 1])
    std[2] += torch.sum(torch.square(images)[:, :, :, 2])

mean /= (len(train_loader.dataset) * 256 * 256)
print(std)
std = torch.sqrt(std / len(train_loader.dataset) / 256 / 256 - mean ** 2)

print(f"Mean: {mean}\nStandard deviation: {std}")