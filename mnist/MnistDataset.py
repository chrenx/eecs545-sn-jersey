import importlib, os
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.io import read_image
import torch


class MnistDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cls_list = sorted(os.listdir(root_dir))
        # self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.cls_list)}
        self.all_img_paths, self.all_imgs = self._get_img_info()
        # self.mean = 0.0
        # self.std = 0.0
        # print("mean: ", str(self.mean), " ,std: ", str(self.std))


    def _get_img_info(self):
        img_paths = []
        imgs_label = []
        # tmp_all = None
        for cls in tqdm(self.cls_list):
            cls_path = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                img_paths.append((img_path, int(cls)))

                img = read_image(img_path)
                if self.transform:
                    img = self.transform(img)
                # if tmp_all is None:
                #     tmp_all = img[None,:,:,:]
                # else:
                #     tmp_all = torch.cat((tmp_all, img[None,:,:,:]), dim=0)
                label = int(cls) - 1
                # if label < 1 or label > 99:
                #     print("有错误")
                #     print(label)
                #     print(cls)
                #     exit(0)
                imgs_label.append((img, label))

        # assert tmp_all.shape[0] == len(imgs_label)
        # self.mean = torch.mean(tmp_all, dim=(0, 2, 3))
        # self.std = torch.std(tmp_all, dim=(0, 2, 3))
        return img_paths, imgs_label

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        # img_path, label = self.all_img_paths[idx]
        # img = read_image(img_path)
        # if self.transform:
        #     img = self.transform(img)
        img, label = self.all_imgs[idx]
        # img; [1, 64, 64]
        return img, label