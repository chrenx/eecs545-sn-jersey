from tqdm import tqdm
import os, cv2
import mat73


gt_train_mat = mat73.loadmat("datasets/train/digitStruct.mat")
# gt_test_mat = mat73.loadmat("datasets/test/digitStruct.mat")

train_img_path = "datasets/train/images"
train_label_path = "datasets/train/labels"
for i in tqdm(range(len(gt_train_mat["digitStruct"]["bbox"]))):
    img_name = gt_train_mat["digitStruct"]["name"][i]  # "1.png"
    output_name = img_name[:-3] + "txt"  # "1.txt"
    labels = gt_train_mat["digitStruct"]["bbox"][i]["label"]
    if not isinstance(labels, list):
        labels = [labels]
    labels = [int(label) for label in labels]  # [1, 9]
    
    img = cv2.imread(os.path.join(train_img_path, img_name))
    H, W, _ = img.shape

    top_list = gt_train_mat["digitStruct"]["bbox"][i]["top"]
    left_list = gt_train_mat["digitStruct"]["bbox"][i]["left"]
    height_list = gt_train_mat["digitStruct"]["bbox"][i]["height"]
    width_list = gt_train_mat["digitStruct"]["bbox"][i]["width"]
    if not isinstance(top_list, list):
        top_list = [top_list]
    if not isinstance(left_list, list):
        left_list = [left_list]
    if not isinstance(height_list, list):
        height_list = [height_list]
    if not isinstance(width_list, list):
        width_list = [width_list]

    with open(os.path.join(train_label_path, output_name), 'w') as f:
        for j in range(len(labels)):
            cls = labels[j]
            top = int(top_list[j])
            left = int(left_list[j])
            height = int(height_list[j])
            width = int(width_list[j])

            # normalized 4 coordinates
            x1, y1 = left / W, top / H
            x2, y2 = (left + width) / W, top / H
            x3, y3 = left / W, (top + height) / H
            x4, y4 = (left + width) / W, (top + height) / H

            if cls == 10:
                cls = 0
            
            f.write(f"{cls} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
        
        # color = (0, 255, 0)  # Green color
        # thickness = 2  # Thickness of the bounding box
        # cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x4 * W), int(y4 * H)), color, thickness)
        # cv2.imwrite('testing123.jpg', img)