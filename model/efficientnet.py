import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

class EfficientNetV2(nn.Module):
    def __init__(self, args):
        super(EfficientNetV2, self).__init__()
        self.model = efficientnet_v2_s()

    def forward(self, x):
        return None