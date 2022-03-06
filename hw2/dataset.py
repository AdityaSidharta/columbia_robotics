import os
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            dataset_dir: string, train_dir, val_dir, and test_dir in segmentation.py.
                         Be careful the images are stored in the subfolders under these directories.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir
        self.has_gt = has_gt

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_rgb,
                                 std=std_rgb)
        ])

        self.rgb_dir = os.path.join(self.dataset_dir, 'rgb')
        self.gt_dir = os.path.join(self.dataset_dir, 'gt')

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError("rgb folder should exist in {}".format(self.dataset_dir))
        else:
            self.rgb_filenames = [os.path.join(self.rgb_dir, x) for x in os.listdir(self.rgb_dir) if x.endswith('.png')]
            self.rgb_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

        if os.path.exists(self.gt_dir):
            self.gt_filenames = [os.path.join(self.gt_dir, x) for x in os.listdir(self.gt_dir) if x.endswith('.png')]
            self.gt_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
            assert len(self.rgb_filenames) == len(self.gt_filenames)
        else:
            self.gt_filenames = None

        self.dataset_length = len(self.rgb_filenames)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        rgb_img = image.read_rgb(self.rgb_filenames[idx])
        rgb_img = self.transform(rgb_img)
        if self.has_gt is False:
            sample = {'input': rgb_img}
        else:
            gt_mask = image.read_mask(self.gt_filenames[idx])
            # print(gt_mask.shape)
            gt_mask = transforms.ToTensor()(gt_mask).type(torch.long).squeeze()
            sample = {'input': rgb_img, 'target': gt_mask}
        return sample
