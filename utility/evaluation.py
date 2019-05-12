#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2019/05/12 06:30:16
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import copy
import json

# import the files of mine
from settings import log
from logger import ImProgressBar
import settings


def evaluate(model, val_loader, device, num_classes, test=True):
    confusion_matrix = np.zeros((num_classes, num_classes))
    # evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pbar = ImProgressBar(len(val_loader))
        for ix, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            y_true = [int(x.cpu().numpy()) for x in targets]
            y_pred = [int(x.cpu().numpy()) for x in predicted]
            for i in range(len(y_true)):
                confusion_matrix[y_true[i], y_pred[i]] += 1

            pbar.update(ix)
        pbar.finish()
        accuracy = correct / total
        # accuracy for each class
        # for i in range(num_classes):
        #     confusion_matrix[i, -2] = np.sum(confusion_matrix[i, :num_classes])
        #     confusion_matrix[i, -1] = confusion_matrix[i, i] / confusion_matrix[i, -2]
        
        log.logger.info('Accuracy on {} set is {}/{} ({:.4f}%)'.format('test ' if test else 'train', correct, total, 100 * accuracy))

        return accuracy, confusion_matrix


def draw_confusion(confusion_training, confusion_validation, epoch):
    # Set up plot
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    cax1 = ax1.matshow(confusion_training)

    ax2 = fig.add_subplot(122)
    cax2 = ax2.matshow(confusion_validation)

    plt.savefig('{}{}.png'.format(settings.DIR_confusion, epoch))

    # Set up axes
    # ax1.set_xticklabels([''] + all_categories, rotation=90)
    # ax1.set_yticklabels([''] + all_categories)
    # ax2.set_xticklabels([''] + all_categories, rotation=90)

    # sphinx_gallery_thumbnail_number = 2
    # plt.show()

def show_curve_1(y1s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    plt.plot(x, y1, label='train')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_2(y1s, y2s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    plt.plot(x, y1, label='train') # train
    plt.plot(x, y2, label='test') # test
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_3(y1s, y2s, y3s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    y3 = np.array(y3s)
    plt.plot(x, y1, label='class0')  # class0
    plt.plot(x, y2, label='class1')  # class1
    plt.plot(x, y3, label='class2')  # class2
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')