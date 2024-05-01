import os, json, glob, torch, heapq, cv2, shutil
from ultralytics import YOLO
from argparse import ArgumentParser
from tqdm import tqdm
import math


def get_coords(coords):
    x1, y1, x2, y2 = None, None, None, None
    for coord in coords:
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
    # x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)
    return [x1, y1, x2, y2]

def get_centroid(coords):
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    # # print(coords)
    # for coord in coords:
    #     if x1 is None:
    #         x1 = coord[0].item()
    #     else:
    #         if coord[0].item() < x1:
    #             x1 = coord[0].item()
    #     if x2 is None:
    #         x2 = coord[0].item()
    #     else:
    #         if coord[0].item() > x2:
    #             x2 = coord[0].item()
    #     if y1 is None:
    #         y1 = coord[1].item()
    #     else:
    #         if coord[1].item() < y1:
    #             y1 = coord[1].item()
    #     if y2 is None:
    #         y2 = coord[1].item()
    #     else:
    #         if coord[1].item() > y2:
    #             y2 = coord[1].item()

    # assert x1 is not None
    # assert y1 is not None 
    # assert x2 is not None
    # assert y2 is not None

    return (x1 + x2) / 2, (y1 + y2) / 2


def recog_num(cls_model, img_path, coords, threshold=0.9):
    img = cv2.imread(img_path)
    x1,y1,x2,y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    img = img[y1:y2, x1:x2]
    res = cls_model.predict(img, verbose=False)
    res = res[0].probs

    if res.top1conf >= threshold:
        return int(res.top1), res.top1conf
    else:
        return None, None


def copy_and_replace(source_path, destination_path):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    shutil.copy2(source_path, destination_path)


def write_to_file(predict_dict, output_pred_dir):
    # write result to file
    with open(output_pred_dir, 'w') as f:
        json.dump(predict_dict, f)


def process_data(obb_model, cls_model, args):
    input_path = args.input_path
    output_pred_dir = f"predict_{args.mode}_orig.json"
    incorrect_pred_dir = f'predict_{args.mode}_incorrect.json'

    img_folder_path = os.path.join(input_path, 'images')
    img_folder_names = os.listdir(img_folder_path)
    tmp = sorted([int(ele) for ele in img_folder_names])
    img_folder_names = [str(ele) for ele in tmp]  # sorted folder name

    # read ground truth if necessary
    if not args.challenge:
        accuracy = 0
        gt_file_path = glob.glob(os.path.join(input_path, '*gt.json'))[0]
        with open(gt_file_path, 'rb') as f:
            gt_dict = json.load(f)

    predict_dict = {}
    incorrect_dict = {}

    if args.proceed:
        with open(args.proceed, 'rb') as f:
            predict_dict = json.load(f)
            print("continue predicting...\n")

    counter = 0
    uncertain_tracklet = set()

    # iterate the images folder
    for img_folder_name in tqdm(img_folder_names):
        print('\n处理tracklet: ', img_folder_name)
        # skip '.DS_store' irrelevant file
        if img_folder_name[0] == '.':
            continue
        if img_folder_name in predict_dict:
            continue

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # img_folder_name = "1"
        img_dir = os.path.join(img_folder_path, img_folder_name)
        
        num_store = [0]*101
        num_conf = [0]*101
        box_center_store = [None]*100
        final_res = None
        two_digit_detected = False # some img with two digits may occluded half

        all_img_names, all_img_paths = [], []
        for tmp in os.listdir(img_dir):
            if tmp[0] == '.':
                continue
            all_img_names.append(tmp)
            all_img_paths.append(os.path.join(img_dir, tmp))

        assert len(all_img_names) == len(all_img_paths)
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        results = obb_model.predict(all_img_paths, save=False, verbose=False, conf=0.65)
        
        # print('当下tracklet参与预测的图片数量: ', len(results))

        # process results of all images in one tracklet 
        for idx, res in enumerate(results):
            # no digits detected
            if len(res.obb.cls) == 0:
                jn = 100
            # one digit detected
            elif len(res.obb.cls) == 1:
                # jn = int(res[0].obb.cls[0].item())
                # print(res.obb.xyxyxyxyn.shape)
                # exit(0)
                coordsn = get_coords(res.obb.xyxyxyxyn[0])  # [x1,y1,x2,y2] normalized
                coords = get_coords(res.obb.xyxyxyxy[0])
                
                centroid_xn, centroid_yn = get_centroid(coordsn)  # normalized centroid
                if centroid_yn < 0.25 or centroid_yn > 0.75:
                    continue 
                jn, conf = recog_num(cls_model, all_img_paths[idx], coords)
                if jn is None:
                    copy_and_replace(all_img_paths[idx], os.path.join(args.uncertain_path, all_img_names[idx]))
                    uncertain_tracklet.add(img_folder_name)
                else:
                    num_store[jn] += 1
                    num_conf[jn] += conf
                    if box_center_store[jn] is None:
                        box_center_store[jn] = centroid_xn
                    else:
                        box_center_store[jn] += centroid_xn

            # two digits detected
            else:
                coordsn_1 = get_coords(res.obb.xyxyxyxyn[0])  # [x1,y1,x2,y2] normalized
                coords_1 = get_coords(res.obb.xyxyxyxy[0])
                coordsn_2 = get_coords(res.obb.xyxyxyxyn[1])  # [x1,y1,x2,y2] normalized
                coords_2 = get_coords(res.obb.xyxyxyxy[1])
                centroid_xn_1, centroid_yn_1 = get_centroid(coordsn_1)
                centroid_xn_2, centroid_yn_2 = get_centroid(coordsn_2)

                jn, conf = None, None

                if centroid_yn_1 < 0.25 or centroid_yn_1 > 0.75:
                    jn_1 = None
                else:
                    jn_1, conf_1 = recog_num(cls_model, all_img_paths[idx], coords_1)
                
                if centroid_yn_2 < 0.25 or centroid_yn_2 > 0.75:
                    jn_2 = None
                else:
                    jn_2, conf_2 = recog_num(cls_model, all_img_paths[idx], coords_2)
                
                if jn_1 is None or jn_2 is None:
                    copy_and_replace(all_img_paths[idx], os.path.join(args.uncertain_path, all_img_names[idx]))
                    uncertain_tracklet.add(img_folder_name)
                    if jn_1 is not None:
                        jn = jn_1
                        conf = conf_1
                    if jn_2 is not None:
                        jn = jn_2
                        conf = conf_2
                else:
                    two_digit_detected = True
                    # print("找到了")
                    # print(len(res.obb.cls))
                    # print(all_img_paths[idx])
                    # print(res.obb)
                    # print()
                    # tttt = cv2.imread(all_img_paths[idx])
                    # cv2.imwrite('123456.jpg', tttt[int(coords_2[1]):int(coords_2[3]), int(coords_2[0]):int(coords_2[2])])
                    # cv2.imwrite('123456+++.jpg', tttt[int(coords_1[1]):int(coords_1[3]), int(coords_1[0]):int(coords_1[2])])

                    if centroid_xn_1 < centroid_xn_2:
                        jn = jn_1 * 10 + jn_2
                    else:
                        jn = jn_2 * 10 + jn_1
                    conf = (conf_1 + conf_2) / 2

                if jn is not None and conf is not None:
                    num_store[jn] += 1
                    num_conf[jn] += conf                     
                            
            # num_store[jn] += 1

            # finish early if possible
            if jn is not None and jn != 100 and num_store[jn] >= args.threshold and \
               two_digit_detected and jn >= 10:
                # just in case most imgs detect single digit when there is
                # only a few has two digits
                final_res = jn
                predict_dict[img_folder_name] = final_res
                write_to_file(predict_dict, output_pred_dir)
                print("two digits threshold达标")
                break


        final_two_digit_res, final_one_digit_res, separate_two = None, None, None

        if two_digit_detected:
            # print("这里3")
            # didn't pass the threshold, but it is ok for now
            # print('max: ', max(num_store[10:100]))
            final_two_digit_res = num_conf[10:100].index(max(num_conf[10:100])) + 10  
            # print("final res: ", final_res)
            # predict_dict[img_folder_name] = final_res
            # write_to_file(predict_dict, output_pred_dir)
        # else:
            # some strategies to process the digits detected
            # two digits: but img catches only one for each
        single_conf = num_conf[:10]
        first_digit, second_digit = heapq.nlargest(2, range(len(single_conf)), 
                                                key=single_conf.__getitem__)
        if num_store[first_digit] == 0:
            final_one_digit_res = -1
        elif num_store[second_digit] == 0:
            final_one_digit_res = first_digit
        else:
            final_one_digit_res = first_digit
            # else:
            #     first_centroid = box_center_store[first_digit] / num_store[first_digit]
            #     second_centroid = box_center_store[second_digit] / num_store[second_digit]
            #     if first_centroid[0] > second_centroid[0]:
            #         separate_two = first_digit * 10 + second_digit
            #     else:
            #         separate_two = second_digit * 10 + first_digit

        # if num_conf[final_two_digit_res] / num_store[final_two_digit_res] > 0.96:
        #     print("可疑")
        #     print(num_conf[final_two_digit_res] / num_store[final_two_digit_res] )
        #     final_res = final_two_digit_res
        # else:
            # 进行2-digit与1-digit的比较
        if final_two_digit_res is None:
            final_res = final_one_digit_res
        elif final_one_digit_res == -1 or final_one_digit_res is None:
            final_res = final_two_digit_res
        else:
            if abs(num_store[final_one_digit_res] - num_store[final_two_digit_res]) / num_store[final_one_digit_res] > 0.9:
                final_res = final_one_digit_res
            else:
                final_res = final_two_digit_res

        print(num_store)



        predict_dict[img_folder_name] = final_res
        write_to_file(predict_dict, output_pred_dir)
        # print("最后")
        # print(num_store)
        print("------------------------")


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
    print("\nUncertain tracklet:")
    print(uncertain_tracklet)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--proceed", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--mode", type=str, default='challenge')
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--obb_model_path", type=str, default='/home/chrenx/eecs545-sn-jersey/yolo-bb/best-03-25.pt')
    parser.add_argument("--cls_model_path", type=str, default='/home/chrenx/eecs545-sn-jersey/yolo-cls/best-cls.pt')

    args, unknown = parser.parse_known_args()

    # args.clean_dir = f'data/jersey-2023-cleaned/{args.mode}/images'
    # args.input_path = f'data/jersey-2023/{args.mode}'
    args.input_path = f'data/jersey-2023/{args.mode}'
    args.uncertain_path = '/home/chrenx/eecs545-sn-jersey/prediction_uncertain'

    obb_model = YOLO(args.obb_model_path)
    cls_model = YOLO(args.cls_model_path)

    process_data(obb_model, cls_model, args)
