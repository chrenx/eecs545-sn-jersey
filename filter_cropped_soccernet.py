import os, json, glob, torch, heapq, shutil, sys
from ultralytics import YOLO
from argparse import ArgumentParser
from tqdm import tqdm
import cv2
import math


def crop_data(model, args):
    input_path = args.input_path

    img_folder_path = os.path.join(input_path, 'images')
    img_folder_names = os.listdir(img_folder_path)
    img_folder_names = sorted([int(i) for i in img_folder_names if i[0] != '.'])

    non_detected_folder = []

    # iterate the images folder
    with tqdm(range(args.start_idx, len(img_folder_names))) as max_len:
        for idx in max_len:
            img_folder_name = img_folder_names[idx]
            # skip '.DS_store' irrelevant file
            # if img_folder_name[0] == '.':
            #     continue

            img_folder_name = str(img_folder_name)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # img_folder_name = "0"
            img_dir = os.path.join(img_folder_path, img_folder_name)
            

            all_img_names = os.listdir(img_dir)
            # all_img_names = sorted(all_img_names)
            all_img_names.sort()
         
            tmp = []
            for img_name in all_img_names:
                if img_name[0] == '.':
                    continue
                if os.path.exists(os.path.join(args.clean_dir, img_name)):
                    continue
                tmp.append(img_name)
            if len(tmp) == 0:
                continue

            all_img_names = tmp
        
            all_img_paths = [os.path.join(img_dir, img_name) for img_name in all_img_names]
            
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            results = model.predict(all_img_paths, save=False, 
                                    conf=args.conf, verbose=False, max_det=2)

            # process results of all images in one tracklet 
            for index, res in enumerate(results):
                if os.path.exists(os.path.join(args.clean_dir, img_folder_name, 
                                               all_img_names[index])):
                    continue
                # no digits detected
                if len(res.obb.cls) == 0:
                    non_detected_folder.append(index)
                    continue
                # digits detected
                x1, y1, x2, y2 = None, None, None, None
                for digit_idx in range(len(res.obb.cls)):
                    for coord in res.obb.xyxyxyxy[digit_idx]:
                        if x1 is None:
                            x1 = coord[0].item()
                        else:
                            if coord[0].item() < x1:
                                x1 = coord[0].item()
                        if x2 is None:
                            x2 = coord[0].item()
                        else:
                            if coord[0].item() > x2:
                                x2 = coord[0].item()
                        if y1 is None:
                            y1 = coord[1].item()
                        else:
                            if coord[1].item() < y1:
                                y1 = coord[1].item()
                        if y2 is None:
                            y2 = coord[1].item()
                        else:
                            if coord[1].item() > y2:
                                y2 = coord[1].item()

                assert x1 is not None
                assert y1 is not None
                assert x2 is not None
                assert y2 is not None
                x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)

                orig_img = cv2.imread(all_img_paths[index])
                cropped_img = orig_img[y1:y2, x1:x2]

                dest = os.path.join(args.clean_dir, img_folder_name)
                os.makedirs(dest, exist_ok=True)
                dest = os.path.join(dest, all_img_names[index])
                try:
                    cv2.imwrite(dest, cropped_img)
                except:
                    non_detected_folder.append(index)
                    continue



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--model_path", type=str, default='yolo-bb/best-03-25.pt')
    parser.add_argument("--start_idx", type=int, default=0)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    args.clean_dir = f'data/jersey-2023-cleaned-crop/{args.mode}/images'
    args.input_path = f'data/jersey-2023-cleaned/{args.mode}'

    model = YOLO(args.model_path)

    crop_data(model, args)
