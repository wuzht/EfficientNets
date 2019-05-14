#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   load_dataset.py
@Time    :   2019/05/12 02:09:44
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# import the files of mine
from settings import log
import settings
from logger import ImProgressBar

def load_class_label():
    """
    class_label = {'n02124075': 0, 'n04067472': 1, 'n04540053': 2, ... }
    """
    class_label = {}
    with open(settings.PATH_class_label) as class_label_file:
        for i, line in enumerate(class_label_file):
            class_label[str(line.rstrip('\n'))] = i
    # print(len(class_label), class_label)
    return class_label


def load_val_info(class_label):
    images = []
    with open(settings.PATH_val_annotations) as val_annotations_file:
        for _, line in enumerate(val_annotations_file):
            line_list = str(line.rstrip('\n')).split()
            images.append({
                "path": settings.DIR_val_set + line_list[0],
                "name": line_list[1],
                "label": class_label[line_list[1]],
                "bbox": [int(x) for x in line_list[2:]],
            })
    
    val = {
        "images": images
    }
    return val


def load_train_info(class_label):
    images = []
    for class_name, label in class_label.items():
        cur_dir = '{}{}/'.format(settings.DIR_train_set, class_name)
        with open('{}{}_boxes.txt'.format(cur_dir, class_name)) as info_file:
            for _, line in enumerate(info_file):
                line_list = str(line.rstrip('\n')).split()
                images.append({
                    "path": cur_dir + 'images/' + line_list[0],
                    "name": class_name,
                    "label": label,
                    "bbox":  [int(x) for x in line_list[1:]]
                })

    train = {
        "images": images
    }
    return train


def init_dataset_info():
    dataset_info = {}

    dataset_info["class_label"] = load_class_label()
    dataset_info["val"] = load_val_info(dataset_info["class_label"])
    dataset_info["train"] = load_train_info(dataset_info["class_label"])
    
    with open(settings.PATH_dataset_info_json, 'w') as json_file:
        json_file.write(json.dumps(dataset_info))

    return dataset_info


def calculate_mean_std(data_loader):
    pop_mean = []
    pop_std = []
    pbar = ImProgressBar(total_iter=len(data_loader))
    for i, (imgs, labels) in enumerate(data_loader):
        np_imgs = imgs.numpy()  # (b, 3, h, w)

        # shape (3,)
        batch_mean = np.mean(np_imgs, axis=(0, 2, 3))
        batch_std = np.std(np_imgs, axis=(0, 2, 3))
        
        pop_mean.append(batch_mean)
        pop_std.append(batch_std)

        pbar.update(i)
    pbar.finish()

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)

    print(pop_mean, pop_std)    # [0.50199103 0.50199103 0.50199103] [0.37681857 0.37681857 0.37681857]
    return pop_mean, pop_std


class TinyImageNetDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.data_type = "train" if train else "val"
        self.transform = transform
        
        with open(settings.PATH_dataset_info_json) as json_file:
            dataset_info = json.load(json_file)
            self.dataset = dataset_info[self.data_type]

    def __getitem__(self, index):
        path = self.dataset["images"][index]["path"]    # './ImageNet/tiny-imagenet-200/train/n02883205/images/n02883205_162.JPEG'
        label = self.dataset["images"][index]["label"]
        # some images in the dataset are 8-bit black and white, have only one channel, so it should perform convert('RGB')
        img = Image.open(path).convert('RGB')  # PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=64x64

        if self.transform != None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset["images"])