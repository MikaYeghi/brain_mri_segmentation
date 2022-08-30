import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import torch.nn as nn
import os

def init_backbone(in_channels, classes, encoder='resnet34', encoder_weights='imagenet', device='cuda', activation='softmax'):
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
        self.preprocessing = get_preprocessing_fn('resnet34', pretrained='imagenet')
        self.device = device
        self.save_path = save_path

    def forward(self, x):
        return self.backbone(x)

    def to(self, device):
        self.backbone = self.backbone.to(device)
    
    def preprocess(self, image):
        image = image.cpu()
        preprocessed_image = self.preprocessing(image).to(self.device)
        return preprocessed_image
    
    def preprocess_batch(self, images_batch):
        for i in range(len(images_batch)):
            images_batch[i] = self.preprocess(images_batch[i])
        return images_batch
    
    def save(self):
        path = os.path.join(self.save_path, "model.pt")
        torch.save(self.backbone.state_dict(), path)
    
    def load(self, path):
        """TO DO"""
        pass