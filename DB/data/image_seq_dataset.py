import functools
import logging
import bisect

import torch.utils.data as data
import cv2
import numpy as np
import glob
from concern.config import Configurable, State
import math
import os
import json

class ImageSeqDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        if 'train' in self.data_dir[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.track_paths = []
        self.gts = []
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)): 
            pth = self.data_dir[i] # /home/yuningc/jersey-2023/train
            files = os.listdir(pth) #[images, train_gt.json]
            for i in files:
                if ".json" in i:
                    with open(os.path.join(pth,i),'r') as f:
                        gt = json.load(f)
                    self.gts += list(gt.values()) #[/home/yuningc/jersey-2023/train/train_gt.json]
            r = pth+'/images/'#  /home/yuningc/jersey-2023/train/images
            all_image_paths = []
            for f in os.listdir(r): # [0,1,...]
                image_list = os.listdir(os.path.join(r,f)) #[img_1.jpg,...]
                image_path=[r+f+"/"+timg.strip() for timg in image_list]
                all_image_paths.append(image_path)
            self.track_paths += all_image_paths
        
        self.gts = (np.array(self.gts)+100)%100
        
        # self.gts[self.gts<100] = 1
        # self.gts[self.gts==100] = 0
        

        self.num_samples = len(self.track_paths)

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        track_path = self.track_paths[index]
        imgs = []
        for image_path in track_path:
            if os.path.exists(image_path):
                img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
                #try:
                #except:
                    #continue
                imgs.append(img)
        '''
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        '''
        data['image'] = imgs
        target = np.array(self.gts[index])
        data['cls'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.track_paths) 
