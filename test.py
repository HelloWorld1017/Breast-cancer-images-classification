from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
from datetime import datetime
from torchvision.datasets import ImageFolder
from torchvision import models
from vit import VisionTransformer

def eval(pred,real):
    #为三类问题
    f=[]
    for i in range(3):
        Sp=0
        TP=0
        FN=0
        for j in range(len(pred)):
            if pred[j] == i: #预测为真
                Sp += 1
                if real[j] == i:
                    TP += 1
            else:
                if real[j] == i:
                    FN += 1
        precision = TP/(Sp+1e-6)
        recall = TP/(TP+FN)
        f1=2*precision*recall/(precision+recall+1e-6)
        f.append(f1)
    return sum(f)/3




