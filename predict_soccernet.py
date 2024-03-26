import os, json, glob
from ultralytics import YOLO
from argparse import ArgumentParser


def process_data(model, args):
    input_path = args.input_path
    output_dir = args.output_dir

    img_folder_path = os.path.join(input_path, 'images')
    img_folder_names = os.listdir(img_folder_path)
    img_folder_names.sort()

    # read ground truth if necessary
    if not args.challenge:
        accuracy = 0
        gt_file_path = glob.glob(os.path.join(input_path, '*gt.json'))[0]
        with open(gt_file_path, 'rb') as f:
            gt_dict = json.load(f)

    predict_dict = {}

    # iterate the images folder
    for img_folder_name in img_folder_names[:4]:
        # skip '.DS_store' irrelevant file
        if img_folder_name[0] == '.':
            continue

        img_dir = os.path.join(img_folder_path, img_folder_name)
        
        # process all images in one tracklet
        num_store = [0]*101
        box_center_store = [None]*100
        final_res = None
        found = False 
        two_digit_detected = False # some img with two digits may occluded half
        all_img_names = os.listdir(img_dir)
        # all_img_names = sorted(all_img_names)
        all_img_names.sort()
       
        all_img_paths = [os.path.join(img_dir, img_name) \
                         for img_name in all_img_names if img_name[0] != '.']
        results = model.predict(all_img_paths[:100], save=False, conf=0.60)
        
        print('当下tracklet参与预测的图片数量: ', len(results))

        for res in results:
            if len(res.obb.cls) == 0:
                jn = 100
            elif len(res.obb.cls) == 1:
                jn = int(res[0].obb.cls[0].item())
                
            else:
                two_digit_detected = True
                first_digit = int(res.obb.cls[0].item())
                second_digit = int(res.obb.cls[1].item())
                jn = first_digit * 10 + second_digit
            
            num_store[jn] += 1

            # finish early if could
            if jn != 100 and num_store[jn] >= args.threshold and two_digit_detected:
                # just in case most imgs detect single digit while there is
                # only a few has two digits
                found = True
                final_res = jn
                break

        if not found:
            if two_digit_detected:
                final_res = num_store.index(max(num_store[10:100]))
            

        if num_store[final_res] < 1:
            final_res = -1

        # write result to file
        predict_dict[img_folder_name] = final_res
        print(predict_dict)

    with open(output_dir, 'w') as f:
        json.dump(predict_dict, f)

    # calculate accuracy
    if not args.challenge:
        count = len(predict_dict.keys())
        correct = 0
        for key, val in predict_dict.items():
            if val == gt_dict[key]:
                correct += 1
        accuracy = correct / count * 100
        print(f"accuracy for {args.mode}: {accuracy}%")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--threshold", type=int, default=70)
    parser.add_argument("--output_dir", type=str, default='./predict.json')
    parser.add_argument("--input_path", type=str, default='data/jersey-2023/train')
    parser.add_argument("--model_path", type=str, default='yolo-bb/best-03-25.pt')
    args = parser.parse_args()

    model = YOLO(args.model_path)

    process_data(model, args)
