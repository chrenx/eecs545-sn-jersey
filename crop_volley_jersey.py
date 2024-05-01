import os, cv2, math
from tqdm import tqdm
from argparse import ArgumentParser


def find_boundary(input_coord):  # 必须是偶数length
    x1, y1, x2, y2 = None, None, None, None
    for idx in range(len(input_coord)):
        ele = input_coord[idx]
        if idx % 2 == 0:  # x
            if x1 is None:
                x1 = ele
            else:
                if ele < x1:
                    x1 = ele
            if x2 is None:
                x2 = ele
            else:
                if ele > x2:
                    x2 = ele
        else:  # y
            if y1 is None:
                y1 = ele
            else:
                if ele < y1:
                    y1 = ele
            if y2 is None:
                y2 = ele
            else:
                if ele > y2:
                    y2 = ele
    return [x1,y1,x2,y2]


def find_closest_digit(idx, digits, centroids, threshold_x=0.35, threshold_y=0.13):
    closest_idx_list = []
    for i in range(len(digits)):
        if i == idx:
            continue
        if math.dist((centroids[idx][0],0), (centroids[i][0],0)) <= threshold_x and \
           math.dist((0,centroids[idx][1]), (0,centroids[i][1])) <= threshold_y:
            closest_idx_list.append(i)
    if len(closest_idx_list) == 0:
        return None
    min_dist = 1
    res = None
    for tmp in closest_idx_list:
        tmp_dist = math.dist(centroids[idx], centroids[tmp])
        if tmp_dist < min_dist:
            min_dist = tmp_dist
            res = tmp
    assert res is not None
    return res


parser = ArgumentParser()
parser.add_argument("--mode", type=str, required=True)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

img_dir = f"yolo-bb/datasets/{args.mode}/images"
label_dir = f"yolo-bb/datasets/{args.mode}/labels"

dest_dir = f"yolo-cls/datasets/{args.mode}"

# cga_cacique_out_1_3_jpg.rf.6078e3faf1b97bde0597f06937e950eb.jpg

for img_name in tqdm(os.listdir(img_dir)):
    if img_name[0] == '.':
        continue

    #!  !!!!!!
    # img_name = "cga_cacique_out_1_3_jpg.rf.6078e3faf1b97bde0597f06937e950eb.jpg"

    orig_img = cv2.imread(os.path.join(img_dir, img_name))
    H, W, _ = orig_img.shape
    label_name = img_name.replace('.jpg', '.txt')
    digits = []
    coords = []
    centroids = []
    x1, y1, x2, y2 = None, None, None, None
    f = open(os.path.join(label_dir, label_name))

    while True:
        content = f.readline()
        if not content:
            break
        content = content.split()
        digits.append(int(content[0]))
        coords.append([float(content[1]), float(content[2]), float(content[3]), 
                       float(content[4]), float(content[5]), float(content[6]),
                       float(content[7]), float(content[8])])
        y = sum([coords[-1][1], coords[-1][3], coords[-1][5], coords[-1][7]]) / 4
        x = sum([coords[-1][0], coords[-1][2], coords[-1][4], coords[-1][6]]) / 4
        centroids.append((x,y))

    f.close()

    label_list = []
    boundary_list = []
    match_set = set()
    if len(digits) == 0:
        continue
    if len(digits) == 1:
        label = str(digits[0])
        x1,y1,x2,y2 = find_boundary(coords[-1])
        label_list.append(label)
        boundary_list.append([x1,y1,x2,y2])
    else:
        for idx, ele in enumerate(digits):
            if idx in match_set:
                continue
            closest_idx = find_closest_digit(idx, digits, centroids)
            if closest_idx is None:
                label_list.append(str(ele))
                boundary_list.append(find_boundary(coords[idx]))

            else:
                match_set.add(closest_idx)
                if centroids[idx][0] < centroids[closest_idx][0]:
                    label_list.append(str(ele*10+digits[closest_idx]))
                else:
                    label_list.append(str(digits[closest_idx]*10+ele))
                concate_list = coords[idx] + coords[closest_idx]
                boundary_list.append(find_boundary(concate_list))
            

    for idx, label in enumerate(label_list):
        x1,y1,x2,y2 = boundary_list[idx]
        cropped_img = orig_img[int(H*y1):int(H*y2), int(W*x1):int(W*x2)]
        dest = os.path.join(dest_dir, label)
        # print(dest)
        os.makedirs(dest, exist_ok=True)
        dest = os.path.join(dest, img_name.replace('.jpg', f'{-idx}.jpg'))
        try:
            cv2.imwrite(dest, cropped_img)
        except:
            continue
