import os, json, glob, torch, heapq, shutil, sys
from ultralytics import YOLO
from argparse import ArgumentParser
from tqdm import tqdm


def filter_data(model, args):
    input_path = args.input_path

    img_folder_path = os.path.join(input_path, 'images')
    img_folder_names = os.listdir(img_folder_path)
    img_folder_names = sorted([int(i) for i in img_folder_names if i[0] != '.'])

    # iterate the images folder
    with tqdm(range(args.start_idx, len(img_folder_names))) as max_len:
        for idx in max_len:
            img_folder_name = img_folder_names[idx]
            # skip '.DS_store' irrelevant file
            # if str(img_folder_name[0]) == '.':
            #     continue

            img_folder_name = str(img_folder_name)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            img_dir = os.path.join(img_folder_path, img_folder_name)

            if os.path.exists(os.path.join(args.clean_dir, img_folder_name)):
                continue

            all_img_names = os.listdir(img_dir)

            # all_img_names = sorted(all_img_names)
            all_img_names.sort()

            all_img_names = [img_name \
                            for img_name in all_img_names if img_name[0] != '.']
        
            all_img_paths = [os.path.join(img_dir, img_name) for img_name in all_img_names]
            
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            results = model.predict(all_img_paths, save=False, 
                                    conf=args.conf, verbose=False, max_det=2)
            

            # process results of all images in one tracklet 
            for index, res in enumerate(results):
                # no digits detected
                if len(res.obb.cls) == 0:
                    continue
                # digits detected
                source_path = all_img_paths[index]
                dest = os.path.join(args.clean_dir, img_folder_name)
                os.makedirs(dest, exist_ok=True)
                dest = os.path.join(dest, all_img_names[index])
                shutil.copyfile(source_path, dest)


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
    args.clean_dir = f'data/jersey-2023-cleaned/{args.mode}/images'
    args.input_path = f'data/jersey-2023/{args.mode}'

    model = YOLO(args.model_path)

    filter_data(model, args)
