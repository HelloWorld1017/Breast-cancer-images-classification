import os
import dataloder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG

import cv2
import numpy as np


# 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
# 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# 利用torch提供的Dataset类，定义我们自己的数据集
class BagDataset(Dataset):

    def __init__(self, transform=None,file_name=None):
        self.transform = transform
        self.file_name=file_name
    def __len__(self):
        return len(os.listdir(self.file_name))
    def __getitem__(self, idx):
        img_name = os.listdir(self.file_name)[idx]
        imgA = cv2.imread(self.file_name + img_name)
        #print(img_name)
        imgA = cv2.resize(imgA, (224, 224))

        # 读取GT
        temp = self.file_name.split('/')
        if temp[-1] == 'BC_IDC_Grade_1':
            GT = torch.FloatTensor([1,0,0])
        elif temp[-1] == 'BC_IDC_Grade_2':
            GT = torch.FloatTensor([0,1,0])
        else:
            GT = torch.FloatTensor([0,0,1])

        if self.transform:
            imgA = self.transform(imgA)

        return imgA, GT, img_name


