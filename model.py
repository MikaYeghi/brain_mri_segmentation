import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import torch.nn as nn
import cv2
import os

import pdb
from matplotlib import pyplot as plt

def init_backbone(in_channels, classes, encoder='vgg11', encoder_weights='imagenet', device='cuda', activation='sigmoid'):
    print(f"Initializing a backbone with {in_channels} input channels, {classes} output classes and {encoder} as the encoder.")
    backbone = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    backbone = backbone.to(device)
    return backbone

class MRIModel(nn.Module):
    def __init__(self, backbone, device='cuda', save_path='saved_models/') -> None:
        super().__init__()
        self.backbone = backbone.to(device)
        # self.preprocessing = get_preprocessing_fn('resnet18', pretrained='imagenet')
        self.device = device
        self.save_path = save_path

        # Preprocessing info
        self.mean = torch.tensor([23.3841, 21.2731, 22.3250], device=device)
        self.std = torch.tensor([34.3730, 31.6041, 32.8448], device=device)

    def forward(self, x):
        res = self.backbone(x)
        res = torch.squeeze(res, 1)
        return res

    def to(self, device):
        self.backbone = self.backbone.to(device)
    
    def preprocess(self, image):
        preprocessed_image = image / 255.
        return preprocessed_image
    
    def preprocess_batch(self, images_batch):
        for i in range(len(images_batch)):
            images_batch[i] = self.preprocess(images_batch[i])
        return images_batch
    
    def save(self):
        """Saves the model to a file."""
        path = os.path.join(self.save_path, "model.pt")
        torch.save(self.backbone.state_dict(), path)
    
    def load(self, path):
        """Loads the model from a file."""
        try:
            self.backbone.load_state_dict(torch.load(path))
            print(f"Model {path} loaded successfully.")
        except Exception as e:
            print(e)