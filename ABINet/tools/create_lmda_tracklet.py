import os
import json

# Function to read bounding box labels from JSON file
def read_labels_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Directory paths
images_dir = "/home/yuningc/jersey-2023/test_crop/images/"
json_file = "/home/yuningc/jersey-2023/test_crop/test_gt.json"
output_file = "/home/yuningc/jersey-2023/test_crop/gt.txt"

# Read labels from JSON file
labels_dict = read_labels_from_json(json_file)

# Open output file for writing
tot = 0
with open(output_file, 'w') as f_out:
    # Iterate through image directories
    for subdir, dirs, files in os.walk(images_dir):
        # Get the number (subdirectory name)
        number = os.path.basename(subdir)
        # Check if the directory contains images
        if files:
            # Iterate through image files
            for file in files:
                if file.endswith('.jpg'):
                    # Image path
                    image_path = os.path.join(subdir, file)
                    # Get labels for current number
                    labels = labels_dict.get(number, [])
                    if labels == -1:
                        # labels = ""
                        tot +=1
                        continue
                    # Write to output file
                    f_out.write(f"{image_path}\t{labels}\n")
print(tot)
