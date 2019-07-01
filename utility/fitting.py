#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fitting.py
@Time    :   2019/05/12 14:38:43
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow as tf
import tfplot
import matplotlib.pyplot as plt

# import the files of mine
import utility.confusion
import utility.evaluation
import utility.save_load
from settings import log
import settings
from logger import ImProgressBar

global_step = 0

def evaluate(model, loader, loss_func, device, num_classes, val=True):
    confusion_matrix = np.zeros((num_classes, num_classes))
    model.eval()
    with torch.no_grad():
        correct = 0
        total_loss = 0
        pbar = ImProgressBar(len(loader))
        for ix, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)

            correct += (predicted == targets).sum().item()
            loss = loss_func(outputs, targets)
            total_loss += loss.item()

            y_true = [int(x.cpu().numpy()) for x in targets]
            y_pred = [int(x.cpu().numpy()) for x in predicted]
            for i in range(len(y_true)):
                confusion_matrix[y_true[i], y_pred[i]] += 1

            pbar.update(ix)
        pbar.finish()
        accuracy = correct / len(loader.dataset)
        average_loss = total_loss / len(loader)

        return average_loss, accuracy, correct, len(loader.dataset), confusion_matrix


def get_lr(cur_step, total_step, base_lr=0.5, warm_up_lr=0.002, lr_decay_rate=0.95):
    warm_up_step_num = total_step // 20
    transition_step_num = total_step // 20
    remain_step_num = total_step - warm_up_step_num - transition_step_num

    if cur_step < warm_up_step_num:
        # 1st stage: constant lr
        return warm_up_lr
    elif cur_step > warm_up_step_num + transition_step_num:
        # 3rd stage: exponential decay
        temp_step = cur_step - warm_up_step_num - transition_step_num
        decay_steps = remain_step_num // 120
        return base_lr * (lr_decay_rate ** (temp_step / decay_steps))
    else:
        # 2nd stage: linearly increase lr from warm_up_lr to base_lr
        transition_delta_lr = (base_lr - warm_up_lr) / transition_step_num
        return (cur_step - warm_up_step_num) * transition_delta_lr + warm_up_lr
    


def train(model, train_loader, loss_func, optimizer, device, total_step_num):
    global global_step
    total_loss = 0
    model.train()
    pbar = ImProgressBar(len(train_loader))
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)         # make predictions
        loss = loss_func(outputs, targets)
        total_loss += loss.item()

        optimizer.zero_grad()   # clear the optimizer before backward
        loss.backward()         # backward
        optimizer.step()        # optimize, update the params of the model

        global_step = global_step + 1
        pbar.update(i)
    pbar.finish()

    return optimizer


def fit(model, num_epochs, optimizer, loss_func, device, train_loader, val_loader, num_classes, total_step_num, lr_decay_period, lr_decay_rate, lr_decay_type="linear"):
    best_val_acc = 0
    best_epoch = 0
    model.to(device)
    loss_func.to(device)

    ######### tensorboard #########
    writer_loss = [tf.summary.FileWriter(settings.DIR_tblog + '/{}_loss/'.format(x)) for x in ['train', 'val']]
    writer_acc = [tf.summary.FileWriter(settings.DIR_tblog + '/{}_acc/'.format(x)) for x in ['train', 'val']]
    writer_lr = tf.summary.FileWriter(settings.DIR_tblog + '/lr/')

    log_var = [tf.Variable(0.0) for i in range(3)]
    tf.summary.scalar('loss', log_var[0])
    tf.summary.scalar('acc', log_var[1])
    tf.summary.scalar('lr', log_var[2])

    write_op = tf.summary.merge_all()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # confusion matrices
    writer_cm_train = tf.summary.FileWriter(os.path.join(settings.DIR_tb_cm, 'train'), session.graph)
    writer_cm_test = tf.summary.FileWriter(os.path.join(settings.DIR_tb_cm, 'test'), session.graph)

    ######### tensorboard #########

    for epoch in range(num_epochs):
        log.logger.info('[Epoch {}/{}] [{}] lr: {}'.format(epoch + 1, num_epochs, settings.now_time, optimizer.param_groups[0]['lr']))

        # train step
        optimizer = train(model, train_loader, loss_func, optimizer, device, total_step_num)

        # evaluate step
        train_loss, train_acc, train_correct, train_total, train_cm = evaluate(model, train_loader, loss_func, device, num_classes, val=False)
        val_loss, val_acc, val_correct, val_total, val_cm = evaluate(model, val_loader, loss_func, device, num_classes, val=True)

        log.logger.info('[Epoch {}/{}] Train set: Avg loss: {:.4f}, Acc: {}/{} ({:.4f}%)'.format(
            epoch+1, num_epochs, train_loss, train_correct, train_total, train_acc * 100
        ))
        log.logger.info('[Epoch {}/{}] Val   set: Avg loss: {:.4f}, Acc: {}/{} ({:.4f}%), best_val_acc: {:.4f}%, epoch: {}'.format(
            epoch+1, num_epochs, val_loss, val_correct, val_total, val_acc * 100, best_val_acc * 100, best_epoch
        ))
        if val_acc > best_val_acc:
            utility.save_load.save_model(model=model, path=settings.PATH_model)
            best_val_acc = val_acc
            best_epoch = epoch

        ######### tensorboard #########
        # loss curve
        losses = [train_loss, val_loss]
        for iw, w in enumerate(writer_loss):
            summary = session.run(write_op, {log_var[0]: float(losses[iw])})
            w.add_summary(summary, epoch)
            w.flush()

        # train and test acc curves
        accs = [train_acc, val_acc]
        for iw, w in enumerate(writer_acc):
            summary = session.run(write_op, {log_var[1]: accs[iw]})
            w.add_summary(summary, epoch)
            w.flush()

        # lr
        summary = session.run(write_op, {log_var[2]: float(optimizer.param_groups[0]['lr'])})
        writer_lr.add_summary(summary, epoch)
        writer_lr.flush()

        # confusion matrices
        figure = utility.confusion.plot_confusion_matrix(train_cm, np.array([str(x) for x in range(num_classes)]))
        if val_acc > best_val_acc:
            plt.savefig('{}tb_train.png'.format(settings.DIR_confusion))
        summary = tfplot.figure.to_summary(figure, tag='train')
        writer_cm_train.add_summary(summary, epoch)
        writer_cm_train.flush()

        figure = utility.confusion.plot_confusion_matrix(val_cm, np.array([str(x) for x in range(num_classes)]))
        if val_acc > best_val_acc:
            plt.savefig('{}tb_test.png'.format(settings.DIR_confusion))
        summary = tfplot.figure.to_summary(figure, tag='test')
        writer_cm_test.add_summary(summary, epoch)
        writer_cm_test.flush()

        plt.close("all")
        ######### tensorboard #########

        # lr decay
        if lr_decay_type == "linear":
            # linear-decay learning rate policy (decreased from lr to 0)
            for param_group in optimizer.param_groups:
                param_group['lr'] -= lr_decay_rate
        elif lr_decay_type == "divide":
            if (epoch + 1) % lr_decay_period == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= lr_decay_rate
        else:
            cur_lr = get_lr(global_step, total_step_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
        
