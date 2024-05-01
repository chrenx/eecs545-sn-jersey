import argparse
import logging
import os
import glob
import tqdm
import torch
import PIL
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils import Config, Logger, CharsetMapper
import json

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def preprocess(img, width, height):
    img = cv2.resize(np.array(img), (width, height))
    img = transforms.ToTensor()(img) #.unsqueeze(0) #Fix: turn on when single
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    return (img-mean[...,None,None]) / std[...,None,None]

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)): 
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)
    
    return pt_text, pt_scores, pt_lengths_

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                        help='path to config file')
    parser.add_argument('--input', type=str, default='figs/test')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default='workdir/train-abinet/best-train-abinet.pth')
    parser.add_argument('--model_eval', type=str, default='alignment', 
                        choices=['alignment', 'vision', 'language'])
    parser.add_argument('--output',type=bool, default=False)
    args = parser.parse_args()
    config = Config(args.config)
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.model_eval is not None: config.model_eval = args.model_eval
    if args.output:
        data = {}
        file_path = os.path.join(args.input,"output_jersey.json")
    config.global_phase = 'test'
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    device = 'cpu' if args.cuda < 0 else f'cuda:{args.cuda}'

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    logging.info(config)

    logging.info('Construct model.')
    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    images = os.path.join(args.input,"images")
    if glob.glob(os.path.join(args.input, '*.json')):
        pth = glob.glob(os.path.join(args.input, '*.json'))[0]
        with open(pth,'r') as f:
            gt = json.load(f)
    acc = 0
    tracklets = [os.path.join(images, fname) for fname in os.listdir(images)]
    tracklets = sorted(tracklets)
    for tracklet in tracklets:
        max_h, max_w = 0,0
        if os.path.isdir(tracklet):
            paths = [os.path.join(tracklet, fname) for fname in os.listdir(tracklet)]
        else:
            paths = glob.glob(os.path.expanduser(args.input))
            assert paths, "The input path(s) was not found"
        paths = sorted(paths)
        numbers = [0]*101
        if len(paths):
            imgs = []
            cnt = 0 
            for path in paths: # tqdm.tqdm(paths):
                img = PIL.Image.open(path).convert('RGB')
                w,h = img.size
                max_h = max(max_h,h)
                max_w = max(max_w,w)
                img = preprocess(img, config.dataset_image_width, config.dataset_image_height)
                img = img.to(device)
                imgs.append(img)
                cnt += 1
                if cnt%100 == 0:
                    imgs = torch.stack(imgs).contiguous().to(device)
                    res = model(imgs)
                    pt_texts, _, __ = postprocess(res, charset, config.model_eval)
                    for pt_text in pt_texts:
                        if pt_text.isdigit():
                            number = int(pt_text)
                            if number <= 0 or number > 99:
                                number = 0
                        else:
                            number = 0
                        numbers[number] +=1
                    imgs = []
            if imgs:
                imgs = torch.stack(imgs).contiguous().to(device)
                res = model(imgs)
                pt_texts, _, __ = postprocess(res, charset, config.model_eval)
                # logging.info(pt_texts)
                for pt_text in pt_texts:
                    # print(pt_text)
                    if pt_text.isdigit():
                        number = int(pt_text)
                        if number <= 0 or number > 99:
                            number = 0
                    else:
                        number = 0
                    # logging.info(number)
                    numbers[number] +=1
            output = int(np.argmax(numbers))
        else:
            output = -1
        if output == 0:
            output = -1
        # if output != -1 and np.max(numbers)/np.sum(numbers) <= 1/3:
            # logging.info(np.max(numbers)/np.sum(numbers))
            # output =  -1
        if len(paths)<=1:# or (len(paths)<=4 and max_h<9 and max_w<8 ): #np.count_nonzero(numbers)>5:
            output = -1
        if args.output:
            data[tracklet.split('/')[-1]] = output
        else:
            if output == gt[tracklet.split('/')[-1]]:
                acc += 1
            else:
                #if gt[tracklet.split('/')[-1]]==-1:
                #    logging.info(f"{np.max(numbers)},{np.count_nonzero(numbers)},{np.max(numbers)/np.count_nonzero(numbers)}")
                logging.info(f"Incorrect {tracklet}: gt {gt[tracklet.split('/')[-1]]} predict {output}")
    if args.output:
        sorted_keys = sorted(map(int, data.keys()))

        # Create a new dictionary with sorted keys
        sorted_data = {str(key): data[str(key)] for key in sorted_keys}
        with open(file_path, "w") as json_file:
            json.dump(sorted_data, json_file)
    else:
        logging.info(f'final acc: {100.0*float(acc)/len(tracklets)}%')
if __name__ == '__main__':
    main()
