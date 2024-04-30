import imgaug
import numpy as np

from concern.config import State
from .data_process import DataProcess
from data.augmenter import AugmenterBuilder
import cv2
import math
from concern.config import Configurable, State
from collections import OrderedDict
import torch


class JerseyData(DataProcess):
    augmenter_args = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.keep_ratio = kwargs.get('keep_ratio')
        self.only_resize = kwargs.get('only_resize')
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        aug = None
        # shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                if len(image)!=3:
                    imgs = []
                    for img in image:
                        imgs.append(self.resize_image(img))
                    data['image'] = np.stack(imgs,axis=0)
                else:
                    data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            # self.may_augment_annotation(aug, data, shape)
        '''
        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        '''
        return data


class AugmentJerseyData(JerseyData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

class JerseyFN(Configurable):
    # padding = State()
    def __init__(self, **kwargs):
        #self.load_all(**kwargs)
        self.padding = 750 

    def __call__(self, batch):
        data_dict = OrderedDict()
        target_height = self.padding
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                if k == 'image':
                    # Pad along the first dimension (height) to achieve a height of 750
                    pad_height = target_height - v.shape[0]
                    if pad_height > 0:
                        # Padding format: (left, right, top, bottom)
                        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 0, 0, pad_height)).contiguous()
                data_dict[k].append(v)
        data_dict['image'] = torch.stack(data_dict['image'], 0)
        data_dict['cls'] = torch.stack(data_dict['cls'], 0)
        
        return data_dict