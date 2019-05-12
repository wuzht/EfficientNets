#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   myshufflenetv2.py
@Time    :   2019/05/12 16:50:59
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x
    

class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BuildingBlock, self).__init__()
        self.stride = stride

        branch_channels = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                # dw
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=self.stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(num_features=in_channels),
                # pw
                nn.Conv2d(in_channels=in_channels, out_channels=branch_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=branch_channels),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            # pw
            nn.Conv2d(in_channels=(in_channels if self.stride > 1 else branch_channels), out_channels=branch_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(in_channels=branch_channels, out_channels=branch_channels, kernel_size=3, stride=self.stride, padding=1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            # pw
            nn.Conv2d(in_channels=branch_channels, out_channels=branch_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride > 1:
            out = torch.cat(tensors=(self.branch1(x), self.branch2(x)), dim=1)
        else:
            # torch.chunk(tensor, chunks, dim=0) → List of Tensors
            # Splits a tensor into a specific number of chunks.
            x1, x2 = x.chunk(chunks=2, dim=1)
            out = torch.cat(tensors=(x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    """
    Implementaion of ShuffleNetV2, construct the ResNet layers with basic blocks (e.g. BuildingBlock)\\
    Args:
        stage_repeat:
        stage_out_channels:
        num_classes: the number of classes
    """
    def __init__(self, stage_repeat, stage_out_channels, in_channels=3, num_classes=200):
        super(ShuffleNetV2, self).__init__()

        self.stage_out_channels = stage_out_channels

        out_channels = self.stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = out_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages stacking
        