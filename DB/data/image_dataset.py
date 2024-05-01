import functools
import logging
import bisect

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
from PIL import Image

def merge_close_bboxes(bboxes, threshold=0.2):
    if not bboxes:
        return bboxes
    bboxes = sorted(bboxes, key=lambda x: x[1])
    merged_bboxes = []
    current_merged_bbox = bboxes[0]

    for bbox in bboxes[1:]:
        class_id1, x_center1, y_center1, width1, height1 = current_merged_bbox
        class_id2, x_center2, y_center2, width2, height2 = bbox

        # Calculate the xmin, xmax, ymin, ymax of both bounding boxes
        xmin1 = x_center1 - width1 / 2
        xmax1 = x_center1 + width1 / 2
        ymin1 = y_center1 - height1 / 2
        ymax1 = y_center1 + height1 / 2

        xmin2 = x_center2 - width2 / 2
        xmax2 = x_center2 + width2 / 2
        ymin2 = y_center2 - height2 / 2
        ymax2 = y_center2 + height2 / 2

        # Calculate the distance between the x_centers of the two bounding boxes
        x_distance = abs(xmax1 - xmin2)+ abs(y_center1-y_center2)

        # Check if the next bbox is horizontally close to the current merged bbox
        if x_distance <= threshold:
            # Calculate the merged bbox's xmin, xmax, ymin, ymax
            xmin = min(xmin1, xmin2)
            xmax = max(xmax1, xmax2)
            ymin = min(ymin1, ymin2)
            ymax = max(ymax1, ymax2)

            # Calculate the merged bbox's x_center, y_center, width, and height
            merged_x_center = (xmin + xmax) / 2
            merged_y_center = (ymin + ymax) / 2
            merged_width = xmax - xmin
            merged_height = ymax - ymin

            # Update the current merged bbox
            current_merged_bbox = (class_id1+class_id2, merged_x_center, merged_y_center, merged_width, merged_height)
        else:
            # Save the current merged bbox
            merged_bboxes.append(current_merged_bbox)
            # Start a new merged bbox
            current_merged_bbox = bbox

    # Save the last merged bbox
    merged_bboxes.append(current_merged_bbox)

    return merged_bboxes


class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                if 'jersey' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()[:-4]+'.txt' for timg in image_list]
                else:
                    gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()+'.txt' for timg in image_list]
            else:
                image_path=[self.data_dir[i]+'/test_images/'+timg.strip() for timg in image_list]
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()+'.txt' for timg in image_list]
                elif 'jersey' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()[:-4]+'.txt' for timg in image_list]
                else:
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt_'+timg.strip().split('.')[0]+'.txt' for timg in image_list]
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def convert(self, x_center_norm, y_center_norm, width_norm, height_norm,img_width,img_height):
        # img_width = 92
        # img_height = 184
        # _, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line)
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height
        x_min = max(0, x_center - width / 2)
        y_min = max(0, y_center - height / 2)
        x_max = min(img_width, x_center + width / 2)
        y_max = min(img_height, y_center + height / 2)
        return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
    
    def load_ann(self):
        res = []
        for image_path,gt in zip(self.image_paths,self.gt_paths):
            lines = []
            reader = open(gt, 'r').readlines()
            bounding_boxes = []
            for line in reader:
                item = {}
                
                parts = line.strip().split(',')
                label = parts[-1]
                if 'jersey' in self.data_dir[0]:
                    data = line.strip().split(' ')
                    class_id = data[0]
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])
                    bounding_boxes.append((class_id, x_center, y_center, width, height))
                if 'TD' in self.data_dir[0] and label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                #     line = self.convert(line)
                if 'icdar' in self.data_dir[0]: # or 'jersey' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif 'jersey' not in self.data_dir[0]:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                if 'jersey' not in self.data_dir[0]:
                    item['poly'] = poly
                    item['text'] = label
                    lines.append(item)
            if 'jersey' in self.data_dir[0]:
                bboxes = merge_close_bboxes(bounding_boxes)
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                for bbox in bboxes:
                    item  = {}
                    label, x_center, y_center, width, height = bbox
                    line = self.convert(x_center,y_center,width,height,img_width,img_height)
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                    item['poly'] = poly
                    item['text'] = label
                    lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        # print(data['image'].shape, data['gt'].shape, data['gt'].max(),data['mask'].shape, data['mask'].max(),data['thresh_mask'].max(), data['thresh_map'].max())
        return data

    def __len__(self):
        return len(self.image_paths)
