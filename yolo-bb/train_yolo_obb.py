from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-cls.yaml')  # build a new model from YAML
model = YOLO('best-03-25.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s-obb.yaml').load('best_1.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='data.yaml', epochs=120, seed=42, save_period=10)