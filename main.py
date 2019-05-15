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
import torchvision.transforms as transforms
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

device = torch.device('cuda:5')
num_classes = 200 

# Hyper parameters
batch_size = 2048
num_epochs = 800
lr = 0.5
# lr_decay_type = "linear"
lr_decay_type = "divide"
lr_decay_period = 40 if lr_decay_type == "divide" else None
lr_decay_rate = 2 if lr_decay_type == "divide" else lr / num_epochs
momentum = 0.9
weight_decay = 1e-3

# Log the preset parameters and hyper parameters
log.logger.critical("Preset parameters:")
log.logger.info('model_name: {}'.format(settings.model_name))
log.logger.info('num_classes: {}'.format(num_classes))
log.logger.info('device: {}'.format(device))

log.logger.critical("Hyper parameters:")
log.logger.info('batch_size: {}'.format(batch_size))
log.logger.info('num_epochs: {}'.format(num_epochs))
log.logger.info('lr: {}'.format(lr))
log.logger.info('lr_decay_type: {}'.format(lr_decay_type))
log.logger.info('lr_decay_period: {}'.format(lr_decay_period))
log.logger.info('lr_decay_rate: {}'.format(lr_decay_rate))
log.logger.info('momentum: {}'.format(momentum))
log.logger.info('weight_decay: {}'.format(weight_decay))

# utility.load_dataset.init_dataset_info()
normalize = transforms.Normalize(mean=[0.50199103, 0.50199103, 0.50199103], std=[0.37681857, 0.37681857, 0.37681857])
train_transform = transforms.Compose([
    # data augmentation
    # transforms.RandomRotation(degrees=[-10, 10]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(size=55),
    # transforms.Resize(size=64),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomResizedCrop(size=64),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64),
    transforms.ToTensor(),
    normalize
])

log.logger.critical("train_transform: \n{}".format(train_transform))
log.logger.critical("val_transform: \n{}".format(val_transform))

train_set = utility.load_dataset.TinyImageNetDataset(train=True, transform=train_transform)
val_set = utility.load_dataset.TinyImageNetDataset(train=False, transform=val_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)

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

def get_model(_model_name, _num_classes):
    if _model_name == 'shufflenetv2_x0_5':
        return models.shufflenetv2.shufflenetv2_x0_5(num_classes=_num_classes)
    elif _model_name == 'shufflenetv2_x1_0':
        return models.shufflenetv2.shufflenetv2_x1_0(num_classes=_num_classes)
    elif _model_name == 'shufflenetv2_x1_5':
        return models.shufflenetv2.shufflenetv2_x1_5(num_classes=_num_classes)
    elif _model_name == 'shufflenetv2_x2_0':
        return models.shufflenetv2.shufflenetv2_x2_0(num_classes=_num_classes)
    elif _model_name == 'imshufflenetv2':
        return models.imshuffle.shufflenetv2()
    elif _model_name == 'resnet34_pre': 
        return models.resnet.resnet34(pretrained=True, num_classes=_num_classes)
    else:
        log.logger.error("model_name error!")
        exit(-1)


# Declare and define the model, optimizer and loss_func
model = get_model(settings.model_name, num_classes)
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(params=model.parameters())
log.logger.critical("optimizer: \n{}".format(optimizer))
log.logger.critical("model: \n{}".format(model))

try:
    log.logger.critical('Start training')
    utility.fitting.fit(model, num_epochs, optimizer, device, train_loader, val_loader, num_classes, lr, lr_decay_period, lr_decay_rate, lr_decay_type=lr_decay_type)
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
    model = utility.save_load.load_model(
        model=get_model(settings.model_name, num_classes),
        path=settings.PATH_model,
        device=device
    )
    utility.evaluation.evaluate(model=model, val_loader=train_loader, device=device, num_classes=num_classes, val=False)
    utility.evaluation.evaluate(model=model, val_loader=val_loader, device=device, num_classes=num_classes, val=True)
    log.logger.info('Finished')
