from turtle import forward
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn as nn

def init_backbone(in_channels, classes, encoder='resnet34', encoder_weights='imagenet', device='cuda'):
    backbone = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    backbone = backbone.to(device)
    return backbone

class MRIModel(nn.Module):
    def __init__(self, backbone, device='cuda') -> None:
        super().__init__()
        self.backbone = backbone.to(device)
        self.preprocessing = get_preprocessing_fn('resnet34', pretrained='imagenet')
        self.device = device

    def forward(self, x):
        return self.backbone(x)

    def to(self, device):
        self.backbone = self.backbone.to(device)
    
    def preprocess(self, image):
        image = image.cpu()
        preprocessed_image = self.preprocessing(image).to(self.device)
        return preprocessed_image