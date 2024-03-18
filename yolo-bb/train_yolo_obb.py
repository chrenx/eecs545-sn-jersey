from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8s-obb.yaml')  # build a new model from YAML
# model = YOLO('yolov8s-obb.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s-obb.yaml').load('yolov8s-obb.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='data.yaml', epochs=100, seed=42, save_period=10, visualize=True)