import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, name="resnet50", d_model=256):
        super().__init__()
        if name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_dim = 512
        else:
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_dim = 2048
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.proj = nn.Conv2d(out_dim, d_model, kernel_size=1)

    def forward(self, x):
        # x: Bx3xHxW
        feat = self.backbone(x)  # BxCxhxw
        feat = self.proj(feat)  # Bxdxhxw
        b, d, h, w = feat.shape
        feat = feat.view(b, d, h * w).permute(0, 2, 1)  # BxNxD
        return feat

