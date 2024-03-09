import torch
from model.efficientnet import EfficientNetV2

from utils.parser import get_args

if __name__ == '__main__':
    args = get_args()
    

    model = EfficientNetV2()
    print(model)