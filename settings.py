#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   settings.py
@Time    :   2019/05/12 01:48:31
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import os
import datetime

# import the files of mine
import logger

## setttings #############################################
# model_name= 'shufflenetv2_x0_5'
# model_name= 'shufflenetv2_x1_0'
# model_name = 'shufflenetv2_x1_5'
model_name = 'shufflenetv2_x2_0'

DIR_dataset = './ImageNet/tiny-imagenet-200/'
DIR_train_set = '{}train/'.format(DIR_dataset)
DIR_val_set = '{}val/images/'.format(DIR_dataset)
PATH_class_label = '{}wnids.txt'.format(DIR_dataset)
PATH_dataset_info_json = './dataset_info.json'
PATH_val_annotations = '{}/val/val_annotations.txt'.format(DIR_dataset)

# folders
DIR_trained_model = './trained_model/'
DIR_logs = './logs/'
DIR_tblogs = './tblogs/'
DIR_confusions = './confusions/'

##########################################################

# create folders
if not os.path.exists(DIR_trained_model):
    os.makedirs(DIR_trained_model)
if not os.path.exists(DIR_logs):
    os.makedirs(DIR_logs)
if not os.path.exists(DIR_tblogs):
    os.makedirs(DIR_tblogs)
if not os.path.exists(DIR_confusions):
    os.makedirs(DIR_confusions)    


now_time = 'T' + datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S') + '='   # now_time: to name the following variables

PATH_model = '{}{}{}.pt'.format(DIR_trained_model, now_time, model_name)     # to save the model
DIR_tblog = '{}{}{}'.format(DIR_tblogs, now_time, model_name)    # tensorboard log
PATH_log = '{}{}{}.log'.format(DIR_logs, now_time, model_name)
DIR_confusion = '{}{}{}/'.format(DIR_confusions, now_time, model_name)
DIR_tb_cm  = os.path.join(DIR_tblog, 'cm')  # confusion matrix

if not os.path.exists(DIR_confusion):
    os.makedirs(DIR_confusion)

log = logger.Logger(PATH_log, level='debug')