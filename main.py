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

# import the files of mine
import settings
from settings import log

import utility.save_load
import utility.fitting
import utility.evaluation
import utility.load_dataset

# import models.resnet
# import models.shufflenetv2
# import models.imshuffle
import models.myshufflenetv2

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

# Hyper parameters
num_classes = 100 if settings.isImageNet100 else 200 
batch_size = 200 if settings.isImageNet100 else 1024
num_epochs = 200 if settings.isImageNet100 else 200
total_step_num = num_epochs * 100000 // batch_size
lr = 0.5
lr_decay_type = "linear"
# lr_decay_type = "divide"
# lr_decay_type = "warmup"
if lr_decay_type == "warmup":
    lr = 0.002
lr_decay_period = 40 if lr_decay_type == "divide" else None
lr_decay_rate = 2 if lr_decay_type == "divide" else lr / num_epochs
momentum = 0.9
weight_decay = 4e-5

# Log the preset parameters and hyper parameters
log.logger.critical("Preset parameters:")
log.logger.info('settings.isImageNet100: {}'.format(settings.isImageNet100))
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

if settings.isImageNet100 is True:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    train_set = utility.load_dataset.ImageNet100(dataset='train', transform=train_transform)
    val_set = utility.load_dataset.ImageNet100(dataset='val', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)

else:
    # utility.load_dataset.init_dataset_info()
    normalize = transforms.Normalize(mean=[0.50199103, 0.50199103, 0.50199103], std=[0.37681857, 0.37681857, 0.37681857])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.ToTensor(),
        normalize
    ])
    train_set = utility.load_dataset.TinyImageNetDataset(train=True, transform=train_transform)
    val_set = utility.load_dataset.TinyImageNetDataset(train=False, transform=val_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    # utility.load_dataset.calculate_mean_std(train_loader)

log.logger.critical("train_transform: \n{}".format(train_transform))
log.logger.critical("val_transform: \n{}".format(val_transform))

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

