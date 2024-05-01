from ultralytics import YOLO
import os
import torch

if torch.cuda.is_available():
    print("CUDA is available ...")

# Load a model
model = YOLO('yolov8s-cls.yaml')  # build a new model from YAML
model = YOLO('yolo-best-03-25.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s-obb.yaml').load('best_1.pt')  # build from YAML and transfer weights


# Train the model
results = model.train(data='yolo_data.yaml', batch=64, epochs=120, seed=42, save_period=10)