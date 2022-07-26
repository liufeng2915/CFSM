
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class SourceTargetFace(Dataset):
    def __init__(self, source_img_path=None, source_list=None, target_img_path=None, target_list=None):

        self.source_img_path = source_img_path
        self.target_img_path = target_img_path
        fid = open(source_list)
        self.source_imgs = fid.read().splitlines()
        fid.close()
        fid = open(target_list)
        self.target_imgs = fid.read().splitlines()
        fid.close()

        self.target_len = len(self.target_imgs)
        self.target_shuf = np.arange(self.target_len)
        np.random.shuffle(self.target_shuf)
        self.target_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(112),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.source_imgs)

    def __getitem__(self, index):
        data = {}
        source_img = cv2.imread(os.path.join(self.source_img_path, self.source_imgs[index]))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.imread(os.path.join(self.target_img_path, self.target_imgs[self.target_shuf[self.target_idx]]))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        self.target_idx += 1
        if self.target_idx >= self.target_len:
            self.target_idx = 0
            np.random.shuffle(self.target_shuf)

        data["target_img"] = self.preproc(target_img)
        data["source_img"] = self.preproc(source_img)
        return data