import os
from PIL import Image
import numpy as np

# Function to read YOLO bounding boxes from a text file
def read_yolo_bounding_boxes(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        bounding_boxes = []
        for line in lines:
            data = line.strip().split(' ')
            class_id = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])
            bounding_boxes.append((class_id, x_center, y_center, width, height))
        return bounding_boxes

# Function to crop image based on bounding box
def crop_image(image, bounding_boxes):
    width, height = image.size
    if(len(bounding_boxes)==0):
        return False
    x_min, y_min, x_max, y_max = width, height, 0, 0
    for class_id, x_center, y_center, box_width, box_height in bounding_boxes:
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int(x1 + box_width * width)
        y2 = int(y1 + box_height * height)
        x_min = min(x_min, x1)
        y_min = min(y_min, y1)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)
    return image.crop((x_min, y_min, x_max, y_max))

# Function to concatenate labels
def concatenate_labels(bounding_boxes):
    sorted_boxes = sorted(bounding_boxes, key=lambda x: x[1])  # Sort bounding boxes based on y-axis position
    labels = [str(class_id) for class_id, _, _, _, _ in sorted_boxes]
    return ''.join(labels)

# Directory paths
images_dir = "/home/yuningc/DB/datasets/jersey/valid/images/"
labels_dir = "/home/yuningc/DB/datasets/jersey/valid/labels/"
output_dir = "/home/yuningc/DB/datasets/jersey/valid/cropped_images/"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through image files
for image_file in os.listdir(images_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))
        
        # Read bounding boxes from label file
        bounding_boxes = read_yolo_bounding_boxes(label_path)
        
        # Open image
        image = Image.open(image_path)
        
        # Crop image based on bounding boxes
        cropped_image = crop_image(image, bounding_boxes)
        
        if cropped_image:
            # Concatenate labels
            labels = concatenate_labels(bounding_boxes)
            
            # Save cropped image
            cropped_image_file = os.path.join(output_dir, image_file)
            cropped_image.save(cropped_image_file)
            
            # Save label information in gt text file
            gt_text_file = os.path.join("/home/yuningc/DB/datasets/jersey/valid/", 'gt.txt')
            with open(gt_text_file, 'a') as f:
                f.write(f"{cropped_image_file}\t{labels}\n")
