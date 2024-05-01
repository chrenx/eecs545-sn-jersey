#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
from tqdm import tqdm 
import json

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Util = Demo(experiment, experiment_args, cmd=args)
    for p in tqdm(os.listdir(args['image_path'])):
        Util.inference(os.path.join(args['image_path'],p), args['visualize'])
    print(Util.tot, Util.labeled_tot)


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']
        self.init_torch_tensor()
        self.model = self.init_model()
        self.resume(self.model, self.model_path)
        self.model.eval()
        self.tot = 0
        self.labeled_tot = 0
        with open('/home/yuningc/jersey-2023/train_filter/train_gt.json','r') as f:
            self.gt = json.load(f)

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        '''
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        '''
        new_width = 64 #128
        new_height = 128 #256
        resized_img = cv2.resize(img, (new_width, new_height))
        # cv2.imwrite(os.path.join(self.args['result_dir'], 'resize.jpg'),resized_img)
        return resized_img
        
    def load_image(self, image_path):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        except:
            return 0,0
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        # img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output, folder):
        batch_boxes, batch_scores = output
        result_dir = os.path.join(self.args['result_dir'], folder)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        for index in range(batch['image'].size(0)):
            flag = False
            final_x_min, final_y_min, final_x_max, final_y_max = None, None, None, None
            filename = batch['filename'][index]
            # result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            # result_file_path = os.path.join(result_dir, result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            
            vis_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype('float32')
            h = vis_image.shape[0]
            for i in range(boxes.shape[0]):
                score = scores[i]
                '''
                if self.gt[folder]==-1 and score < self.args['box_thresh']:
                    continue
                if self.gt[folder]!=-1 and score < self.args['box_thresh']*1.25:
                    continue
                '''
                if score < self.args['box_thresh']:
                    continue
                # flag = True
                # break
                
                box = boxes[i,:,:].reshape(-1).tolist()
                x_min, y_min, x_max, _, _, y_max, _, _ = box
                if y_min < 0.1*h or y_max > 0.9*h:
                    continue
                # if x_max - x_min < 9 and y_max - y_min < 11:
                #    continue
                cropped_image = vis_image[int(y_min):int(y_max),int(x_min):int(x_max)]
                # print(x_min,x_max,y_min,y_max)
                if not os.path.isfile(os.path.join(result_dir, filename.split('/')[-1].split('.')[0]+'_'+str(i)+'.jpg')) and cropped_image.size:
                    cv2.imwrite(os.path.join(result_dir, filename.split('/')[-1].split('.')[0]+'_'+str(i)+'.jpg'), cropped_image) #+'_'+str(i)
                    # print(os.path.join("./test", filename.split('/')[-1].split('.')[0]+'_'+str(i)+'.jpg'))
                    # cv2.imwrite(os.path.join("./test", filename.split('/')[-1].split('.')[0]+'_'+str(i)+'.jpg'), cropped_image) #+'_'+str(i)
                '''
                y_mean = (y_min+y_max)/2
                
                if y_mean < 0.2*h or y_mean > 0.8*h:
                    continue
    
                x_min *= 0.95
                y_min *= 0.95
                x_max *= 1.05
                y_max *= 1.05
                if not final_x_max:
                    final_x_max = x_max
                    final_x_min = x_min
                    final_y_max = y_max
                    final_y_min = y_min
                final_y_min = min(final_y_min,y_min)
                final_x_min = min(final_x_min,x_min)
                final_x_max = max(final_x_max,x_max)
                final_y_max = max(final_y_max,y_max)

              
            if final_y_max:
                vis_image = vis_image[int(final_y_min):int(final_y_max),int(final_x_min):int(final_x_max)]
                cv2.imwrite(os.path.join(result_dir, filename.split('/')[-1].split('.')[0]+'.jpg'), vis_image) #+'_'+str(i)
            '''
            # if flag:
            #    cv2.imwrite(os.path.join(result_dir, filename.split('/')[-1].split('.')[0]+'.jpg'), vis_image) #+'_'+str(i)
        '''  
        if self.gt[result_dir.split('/')[-1]]!=-1 and not flag:
            print(result_dir)
            self.tot += 1
        self.labeled_tot += (self.gt[result_dir.split('/')[-1]]!=-1)
        '''
        
    def inference(self, folder_path, visualize=False):
        # if os.path.isdir(os.path.join(self.args['result_dir'],folder_path.split('/')[-1])):
        #    return 
        all_matircs = {}
        model = self.model
        batch = dict()
        batch['filename'] = []
        batch['shape'] = []
        imgs = []
        for f in os.listdir(folder_path):
            image_path =os.path.join(folder_path,f)
            img, original_shape  = self.load_image(image_path)
            if original_shape == 0:
                continue
            batch['filename'].append(image_path)
            imgs.append(img)
            batch['shape'].append(original_shape)
        
        if not os.path.isdir(os.path.join(self.args['result_dir'],folder_path.split('/')[-1])):
            os.mkdir(os.path.join(self.args['result_dir'],folder_path.split('/')[-1]))
        if not imgs:
            return
        imgs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().contiguous()
        with torch.no_grad():
            batch['image'] = imgs
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            self.format_output(batch, output, folder_path.split('/')[-1])
        


if __name__ == '__main__':
    main()
'''
import os

import torch
import numpy as np
from tqdm import tqdm
import cv2

from experiment import Experiment
from data.data_loader import DistributedSampler


class Localization:
    def __init__(self, experiment: Experiment):
        self.init_device()

        self.experiment = experiment
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.current_lr = 0

        self.total = 0

        self.result_dir = experiment.validation.result_dir
        self.box_thresh = experiment.validation.box_thresh

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)

        return model
    
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            filename = batch['full_filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1]
            result_file_path = os.path.join(self.result_dir, result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            self.logger.info("File name "+ filename)
            for i in range(boxes.shape[0]):
                score = scores[i]
                self.logger.info(score)
                self.logger.info(os.path.join(self.result_dir, filename.split('/')[-1].split('.')[0]+'_'+str(i)+'.jpg'))
                if score < self.box_thresh:
                    continue
                box = boxes[i,:,:].reshape(-1).tolist()
                x_min, y_min, x_max, _, _, y_max, _, _ = box
                vis_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype('float32')
                vis_image = vis_image[int(y_min):int(y_max),int(x_min):int(x_max)]
                cv2.imwrite(os.path.join(self.result_dir, filename.split('/')[-1].split('.')[0]+'_'+str(i)+'.jpg'), vis_image)
                

    def localization(self):
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        model = self.init_model()
        
        validation_loaders = self.experiment.validation.data_loaders

        self.steps = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta


        self.logger.report_time('Init')

        model.eval()
        
        self.validate(validation_loaders, model, epoch, self.steps)
        

    def validate(self, validation_loaders, model, epoch, step):
        model.eval()
        for _, loader in validation_loaders.items():
            self.validate_step(loader, model, False)
    
    def validate_step(self, data_loader, model, visualize=False):
        for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=False)
            if not os.path.isdir(self.result_dir):
                os.mkdir(self.result_dir)
            self.format_output(batch, output)

    def to_np(self, x):
        return x.cpu().data.numpy()
    
'''