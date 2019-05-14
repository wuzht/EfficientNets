#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fitting.py
@Time    :   2019/05/12 14:38:43
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''


import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import tensorflow as tf

# import the files of mine
from settings import log
import settings
import utility.evaluation



def train(model, train_loader, loss_func, optimizer, device):
    """
    train model using loss_fn and optimizer in an epoch.\\
    model: CNN networks\\
    train_loader: a Dataloader object with training data\\
    loss_func: loss function\\
    device: train on cpu or gpu device
    """
    total_loss = 0
    model.train()
    
    # train the model using minibatch
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        loss = loss_func(outputs, targets)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # every 10 iteration, print loss
        if (i + 1) % 10 == 0 or i + 1 == len(train_loader):
            log.logger.info("Step [{}/{}] Train Loss: {}".format(i+1, len(train_loader), loss.item()))


    return total_loss / len(train_loader)


def fit(model, num_epochs, optimizer, device, train_loader, val_loader, num_classes, lr, lr_decay_period, lr_decay_rate, lr_decay_type="linear"):
    loss_func = nn.CrossEntropyLoss()
    model.to(device)
    loss_func.to(device)

    ######### tensorboard #########
    writer_loss = tf.summary.FileWriter(settings.DIR_tblog + '/train_loss/')
    writer_acc = [tf.summary.FileWriter(settings.DIR_tblog + '/train/'), tf.summary.FileWriter(settings.DIR_tblog + '/val/')]

    log_var = [tf.Variable(0.0) for i in range(2)]
    tf.summary.scalar('train loss', log_var[0])
    tf.summary.scalar('acc', log_var[1])

    write_op = tf.summary.merge_all()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    ######### tensorboard #########


    losses = []
    for epoch in range(num_epochs):
        log.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        log.logger.info('Average train loss in this epoch: {}'.format(loss))

        # evaluate step
        train_accuracy, confusion_training = utility.evaluation.evaluate(model, train_loader, device, num_classes, val=False)
        val_accuracy, confusion_validation = utility.evaluation.evaluate(model, val_loader, device, num_classes, val=True)

        utility.evaluation.draw_confusion(confusion_training, confusion_validation, epoch + 1)

        # lr decay
        if lr_decay_type == "linear":
            # linear-decay learning rate policy (decreased from lr to 0)
            for param_group in optimizer.param_groups:
                param_group['lr'] -= lr_decay_rate
        else:
            if (epoch + 1) % lr_decay_period == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= lr_decay_rate


        ######### tensorboard #########
        # train loss curve
        summary = session.run(write_op, {log_var[0]: float(loss)})
        writer_loss.add_summary(summary, epoch)
        writer_loss.flush()

        # train and test acc curves
        accs = [train_accuracy, val_accuracy]
        for iw, w in enumerate(writer_acc):
            summary = session.run(write_op, {log_var[1]: accs[iw]})
            w.add_summary(summary, epoch)
            w.flush()
        ######### tensorboard #########
