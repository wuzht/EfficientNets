#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2019/05/12 02:03:09
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import os
os.environ['KMP_WARNINGS'] = 'off'

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

def choose_gpu():
    """
    return the id of the gpu with the most memory
    """
    # query GPU memory and save the result in `tmp`
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    # read the file `tmp` to get a gpu memory list
    memory_gpu = [int(x.split()[2]) for x in open('tmp','r').readlines()]
    log.logger.info('memory_gpu: {}'.format(memory_gpu))
    # get the id of the gpu with the most memory
    gpu_id = str(np.argmax(memory_gpu))
    # remove the file `tmp`
    os.system('rm tmp')
    return gpu_id

device = torch.device('cuda:{}'.format(choose_gpu()))
num_classes = 100 

# Hyper parameters
batch_size = 200
num_epochs = 200
total_step_num = num_epochs * 100000 // batch_size
lr = 0.5
# lr_decay_type = "linear"
# lr_decay_type = "divide"
lr_decay_type = "warmup"
lr_decay_period = 40 if lr_decay_type == "divide" else None
lr_decay_rate = 2 if lr_decay_type == "divide" else lr / num_epochs
momentum = 0.9
weight_decay = 4e-5

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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
val_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

log.logger.critical("train_transform: \n{}".format(train_transform))
log.logger.critical("val_transform: \n{}".format(val_transform))

train_set = utility.load_dataset.ImageNet100(dataset='train', transform=train_transform)
val_set = utility.load_dataset.ImageNet100(dataset='val', transform=val_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)


def get_model(_model_name, _num_classes):
    if _model_name == 'shufflenetv2_x0_5':
        return models.myshufflenetv2.shufflenetv2_x0_5(num_classes=_num_classes)
    elif _model_name == 'shufflenetv2_x1_0':
        return models.myshufflenetv2.shufflenetv2_x1_0(num_classes=_num_classes)
    elif _model_name == 'shufflenetv2_x1_5':
        return models.myshufflenetv2.shufflenetv2_x1_5(num_classes=_num_classes)
    elif _model_name == 'shufflenetv2_x2_0':
        return models.myshufflenetv2.shufflenetv2_x2_0(num_classes=_num_classes)
    else:
        log.logger.error("model_name error!")
        exit(-1)


# Declare and define the model, optimizer and loss_func
model = get_model(settings.model_name, num_classes)
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(params=model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
log.logger.critical("optimizer: \n{}".format(optimizer))
log.logger.critical("loss_func: \n{}".format(loss_func))
log.logger.critical("model: \n{}".format(model))
    
log.logger.critical('Start training')
utility.fitting.fit(model, num_epochs, optimizer, loss_func, device, train_loader, val_loader, num_classes, total_step_num, lr_decay_period, lr_decay_rate, lr_decay_type=lr_decay_type)
