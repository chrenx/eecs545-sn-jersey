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

class ImageJerDataset(data.Dataset, Configurable):
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
        self.base_image_p = os.path.join(self.data_dir[0],"images")
        self.base_label_p = os.path.join(self.data_dir[0],"labels")
        self.images = os.listdir(self.base_image_p)
        self.labels = os.listdir(self.base_label_p)

    def __getitem__(self, index):
        data = {}
        data['image'] = cv2.imread(os.path.join(self.base_image_p,self.images[index]), cv2.IMREAD_COLOR).astype('float32')
        base = 0
        with open(os.path.join(self.base_label_p,self.labels[index]), "r") as f:
            c = f.readlines()
            for i in range(min(len(c),2)):
                base= base*10+(int(c[i][0]))

        data['cls'] = np.array(base)
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.images) 
