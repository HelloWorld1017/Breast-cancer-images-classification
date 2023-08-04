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
from vit import vit_base_patch16_224,backmodel,MyNet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from test import eval
epoch = 20
batch_size = 4
save_epo = 1
model_choose = 'mix'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#加上transforms
normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
transform=transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])
train_data=ImageFolder('datasets40/train/',transform=transform)
test_data=ImageFolder('datasets40/test/',transform=transform)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)


if model_choose == 'resnet':
    model_name = 'resnet50.pth'
    net = models.resnet50(pretrained=False)
    pre = torch.load(model_name)
    net.load_state_dict(pre)
    net.fc = nn.Linear(2048, 3, True)
    net = net.to(device)
elif model_choose == 'vit':
    model_name = 'vit_base_patch16_224.pth'
    net = vit_base_patch16_224()
    pre = torch.load('vit_base_patch16_224.pth')
    net.load_state_dict(pre)
    net.head = nn.Linear(768, 3, True)
    # net = vit_base_patch224_4032()
    net = net.to(device)
elif model_choose == 'mix':
    model_name='mix.pth'
    # pre = torch.load('model/mix_model_9.pth')
    net = MyNet()
    net = net.to(device)
    # net.load_state_dict(pre)



print(device)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-5)

# 记录训练过程相关指标
all_train_iter_loss = []
all_train_iter_acc = []
all_test_iter_loss = []
all_test_iter_acc = []
# start timing
prev_time = datetime.now()

for epo in range(epoch):
    train_loss = 0
    train_acc = 0
    net.train()
    for index, (bag, bag_msk) in enumerate(train_dataloader):
        # 一张图一张图的处理
        # 将图像分割为36个patch
        bag = bag.to(device)
        bag_msk = bag_msk.to(device)

        optimizer.zero_grad()
        output = net(bag)

        output_np_float = output.cpu().detach().numpy().copy()
        output_np = np.argmax(output_np_float, axis=1)
        bag_msk_np = bag_msk.cpu().detach().numpy().copy()
        train_acc += np.sum((output_np == bag_msk_np))

        loss = criterion(output, bag_msk)
        loss.backward()  # 需要计算导数，则调用backward
        iter_loss = loss.item()  # .item()返回一个具体的值，一般用于loss和acc
        all_train_iter_loss.append(iter_loss)
        train_loss += iter_loss
        optimizer.step()
        if np.mod(index, 15) == 0:
            print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
        # break
    all_train_iter_acc.append(train_acc / len(train_data))

    test_loss = 0
    net.eval()
    real=[]
    pred=[]
    test_acc = 0
    with torch.no_grad():
        for index, (bag, bag_msk) in enumerate(test_dataloader):
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)
            optimizer.zero_grad()
            output = net(bag)

            output_np_float = output.cpu().detach().numpy().copy()
            output_np = np.argmax(output_np_float, axis=1)
            pred.append(int(output_np))

            bag_msk_np = bag_msk.cpu().detach().numpy().copy()
            real.append(int(bag_msk_np))

            test_acc += np.sum((output_np == bag_msk_np))

            loss = criterion(output, bag_msk)
            iter_loss = loss.item()
            all_test_iter_loss.append(iter_loss)
            test_loss += iter_loss
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    prev_time = cur_time

    f1=eval(pred,real)

    print('<---------------------------------------------------->')
    print('epoch: %f' % epo)
    print('%s' % time_str)
    print('epoch train loss = %f, epoch train acc = %f'
          % (train_loss / len(train_dataloader), train_acc / len(train_data)))
    print('epoch test loss = %f, epoch test acc = %f, epoch test f1measure = %f'
          % (test_loss / len(test_dataloader), test_acc / len(test_data), f1))
    print('<---------------------------------------------------->')

    if np.mod(epo, save_epo) == 0:
        # 只存储模型参数
        name = model_name.split('.')[0]
        torch.save(net.state_dict(), 'model/'+name+'_model_{}.pth'.format(epo))
        stict = 'model/'+name+'_model_{}.pth'.format(epo)
        print(stict)
        print('saveing .model/'+name+'_model_{}.pth'.format(epo))