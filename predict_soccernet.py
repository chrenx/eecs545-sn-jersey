import os, json, glob, torch, heapq
from ultralytics import YOLO
from argparse import ArgumentParser


def process_data(model, args):
    input_path = args.input_path
    output_pred_dir = f"predict_{args.mode}.json"
    incorrect_pred_dir = f'predict_{args.mode}_incorrect.json'

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
    incorrect_dict = {}

    # iterate the images folder
    for img_folder_name in img_folder_names[:4]:
        # skip '.DS_store' irrelevant file
        if img_folder_name[0] == '.':
            continue

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        img_folder_name = "1"
        img_dir = os.path.join(img_folder_path, img_folder_name)
        
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
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        results = model.predict(all_img_paths[-100:], save=False, 
                                conf=args.conf, verbose=False)
        
        print('当下tracklet参与预测的图片数量: ', len(results))

        # process results of all images in one tracklet 
        for idx, res in enumerate(results):
            # no digits detected
            if len(res.obb.cls) == 0:
                jn = 100
            # one digit detected
            elif len(res.obb.cls) == 1:
                jn = int(res[0].obb.cls[0].item())
                
            # two digits detected
            else:
                two_digit_detected = True
                print('predicted: ', res.obb.cls)
                print('path: ', all_img_paths[-100+idx])
                first_digit = int(res.obb.cls[0].item())
                second_digit = int(res.obb.cls[1].item())
                jn = first_digit * 10 + second_digit
            
            if len(res.obb.cls) != 0:
                x = torch.Tensor([p[0] for p in res.obb.xyxyxyxyn[0,:]])
                y = torch.Tensor([p[1] for p in res.obb.xyxyxyxyn[0,:]])
                centroid = ((torch.sum(x) / 4).item(), 
                            (torch.sum(y) / 4).item())
                box_center_store[jn] = centroid
                            
            num_store[jn] += 1

            # finish early if possible
            if jn != 100 and num_store[jn] >= args.threshold and \
               two_digit_detected and jn >= 10:
                # just in case most imgs detect single digit when there is
                # only a few has two digits
                found = True
                final_res = jn
                predict_dict[img_folder_name] = final_res
                print("这里1")
                break

        if found:
            # process next folder
            print("这里2")
            break
        else:
            if two_digit_detected:
                print("这里3")
                # didn't pass the threshold, but it is ok for now
                print('max: ', max(num_store[10:100]))
                final_res = num_store[10:100].index(max(num_store[10:100])) + 10  
                print("final res: ", final_res)
                predict_dict[img_folder_name] = final_res
                # process next folder
                break
            
        # some strategies to process the digits detected
        # two digits: but img catches only one for each
        single_digit = num_store[:10]
        first_digit, second_digit = heapq.nlargest(2, range(len(single_digit)), 
                                                   key=single_digit.__getitem__)
        first_centroid = box_center_store[first_digit]
        second_centroid = box_center_store[second_digit]
        if first_centroid[0] > second_centroid[0]:
            final_res = first_digit * 10 + second_digit
        else:
            final_res = second_digit * 10 + first_digit


        predict_dict[img_folder_name] = final_res
        print(predict_dict)

        return ##

    # calculate accuracy
    if not args.challenge:
        count = len(predict_dict.keys())
        correct = 0
        for key, val in predict_dict.items():
            if val == gt_dict[key]:
                correct += 1
            else:
                incorrect_dict[key] = val
        accuracy = correct / count * 100
        print(f"accuracy for {args.mode}: {accuracy}%")

        with open(incorrect_pred_dir, 'w') as f:
            json.dump(incorrect_dict, f)

    # write result to file
    with open(output_pred_dir, 'w') as f:
        json.dump(predict_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--input_path", type=str, default='data/jersey-2023/train')
    parser.add_argument("--model_path", type=str, default='yolo-bb/best-03-25.pt')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    model = YOLO(args.model_path)

    process_data(model, args)
