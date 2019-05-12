#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/05/12 02:03:09
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''


import torch
import torch.utils.data as Data
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image
import random
import numpy as np
import torchvision.models

# import the files of mine
import settings
from settings import log

import utility.save_load
import utility.fitting
import utility.evaluation
import utility.load_dataset

import models.resnet
import models.shufflenetv2
import models.imshuffle

# Device configuration, cpu, cuda:0/1/2/3 available
device = torch.device('cuda:5')
num_classes = 200 

# Hyper parameters
batch_size = 1024
num_epochs = 200
lr = 0.1
momentum = 0.9
weight_decay = 4e-5

# Log the preset parameters and hyper parameters
log.logger.info("Preset parameters:")
log.logger.info('model_name: {}'.format(settings.model_name))
log.logger.info('num_classes: {}'.format(num_classes))
log.logger.info('device: {}'.format(device))

log.logger.info("Hyper parameters:")
log.logger.info('batch_size: {}'.format(batch_size))
log.logger.info('num_epochs: {}'.format(num_epochs))
log.logger.info('lr: {}'.format(lr))
log.logger.info('momentum: {}'.format(momentum))
log.logger.info('weight_decay: {}'.format(weight_decay))

# utility.load_dataset.init_dataset_info()
transform = transforms.Compose([
    # data augmentation
    # transforms.RandomRotation(degrees=[-10, 10]),
    # transforms.RandomCrop(size=384)
    # transforms.RandomHorizontalFlip(p=0.5)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.50199103, 0.50199103, 0.50199103],
        std=[0.37681857, 0.37681857, 0.37681857]
    )
])
train_set = utility.load_dataset.TinyImageNetDataset(train=True, transform=transform)
test_set = utility.load_dataset.TinyImageNetDataset(train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# utility.load_dataset.calculate_mean_std(train_loader)


def checkImage(num=5):
    for _ in range(num):
        img_index = random.randint(1, 10000)
        # print(train_set[img_index])
        img = train_set[img_index][0]
        print(img.shape)
        # print(img)
        # img.save('./{}.JPEG'.format(img_index))
        # np_img = train_set[img_index][0][0].numpy()
        # pil_image = Image.fromarray(np_img) # 数据格式为(h, w, c)
        # print(pil_image)
        # plt.imshow(np_img, cmap='gray')
        # plt.show()
        
    exit()
# checkImage(5)


# Declare and define the model, optimizer and loss_func
model = None

if settings.model_name == 'shufflenetv2_x0_5':
    model = models.shufflenetv2.shufflenetv2_x0_5(num_classes=200)
elif settings.model_name == 'shufflenetv2_x1_5':
    model = models.shufflenetv2.shufflenetv2_x1_5(num_classes=200)
elif settings.model_name == 'shufflenetv2_x2_0':
    model = models.shufflenetv2.shufflenetv2_x2_0(num_classes=200)
elif settings.model_name == 'imshufflenetv2':
    model = models.imshuffle.shufflenetv2()
else:
    model = models.resnet.resnet34(pretrained=True, num_classes=200)

# optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params=model.parameters())
log.logger.info(model)

try:
    log.logger.critical('Start training')
    utility.fitting.fit(model, num_epochs, optimizer, device, train_loader, test_loader, num_classes, lr)
except KeyboardInterrupt as e:
    log.logger.error('KeyboardInterrupt: {}'.format(e))
except Exception as e:
    log.logger.error('Exception: {}'.format(e))
finally:
    log.logger.info("Train finished")
    utility.save_load.save_model(
        model=model,
        path=settings.PATH_model
    )
    temp_model = None
    if settings.model_name == 'shufflenetv2_x0_5':
        model = models.shufflenetv2.shufflenetv2_x0_5(num_classes=200)
    elif settings.model_name == 'shufflenetv2_x1_5':
        temp_model = models.shufflenetv2.shufflenetv2_x1_5(num_classes=200)
    elif settings.model_name == 'shufflenetv2_x2_0':
        temp_model = models.shufflenetv2.shufflenetv2_x2_0(num_classes=200)
    elif settings.model_name == 'imshufflenetv2':
        temp_model = models.imshuffle.shufflenetv2()
    else:
        temp_model = models.resnet.resnet34(pretrained=True, num_classes=200)
    model = utility.save_load.load_model(
        model=temp_model,
        path=settings.PATH_model,
        device=device
    )
    utility.evaluation.evaluate(model=model, val_loader=train_loader, device=device, num_classes=num_classes, test=False)
    utility.evaluation.evaluate(model=model, val_loader=test_loader, device=device, num_classes=num_classes, test=True)
    log.logger.info('Finished')
