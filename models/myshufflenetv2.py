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
    # divide the channels into groups
    channels_per_group = num_channels // groups
    # channel shuffle:
    # 1. reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 2. transpose at dim (groups, channels_per_group)
    x = torch.transpose(x, 1, 2).contiguous()
    # 3. flatten the dim (groups, channels_per_group)
    x = x.view(batch_size, -1, height, width)

    return x
    

class BuildingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BuildingBlock, self).__init__()
        self.stride = stride

        branch_channels = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                # dw 3x3 DWConv
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=self.stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(num_features=in_channels),
                # pw 1x1 Conv
                nn.Conv2d(in_channels=in_channels, out_channels=branch_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=branch_channels),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            # pw 1x1 Conv
            nn.Conv2d(in_channels=(in_channels if self.stride > 1 else branch_channels), out_channels=branch_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            nn.ReLU(inplace=True),
            # dw 3x3 DWConv
            nn.Conv2d(in_channels=branch_channels, out_channels=branch_channels, kernel_size=3, stride=self.stride, padding=1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            # pw 1x1 Conv
            nn.Conv2d(in_channels=branch_channels, out_channels=branch_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride > 1:
            # Concat:
            out = torch.cat(tensors=(self.branch1(x), self.branch2(x)), dim=1)
        else:
            # Channel Split:
            x1, x2 = x.chunk(chunks=2, dim=1)   # torch.chunk(tensor, chunks, dim=0) → List of Tensors. Splits a tensor into a specific number of chunks.
            # Concat:
            out = torch.cat(tensors=(x1, self.branch2(x2)), dim=1)

        # Channel Shuffle
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
    def __init__(self, stage_repeats, stage_out_channels, in_channels=3, num_classes=200):
        super(ShuffleNetV2, self).__init__()
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels

        # stage 1: Conv1 and MaxPool
        out_channels = self.stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = out_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage 2-4: stages stacking
        for i in range(len(self.stage_repeats)):
            stage_name = 'stage{}'.format(i + 2)
            repeats = self.stage_repeats[i]
            out_channels = self.stage_out_channels[i + 1]

            seq = [BuildingBlock(in_channels, out_channels, stride=2)]
            for _ in range(repeats - 1):
                seq.append(BuildingBlock(out_channels, out_channels, stride=1))
            setattr(self, stage_name, nn.Sequential(*seq))
            
            in_channels = out_channels

        # stage 5: Conv5
        out_channels = self.stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        # global avg pooling

        # fc
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global avg pooling
        x = self.fc(x)
        return x


def shufflenetv2_x0_5(in_channels=3, num_classes=200):
    return ShuffleNetV2(
        stage_repeats=[4, 8, 4],
        stage_out_channels=[24, 48, 96, 192, 1024],
        in_channels=in_channels,
        num_classes=num_classes
    )


def shufflenetv2_x1_0(in_channels=3, num_classes=200):
    return ShuffleNetV2(
        stage_repeats=[4, 8, 4],
        stage_out_channels=[24, 116, 232, 464, 1024],
        in_channels=in_channels,
        num_classes=num_classes
    )


def shufflenetv2_x1_5(in_channels=3, num_classes=200):
    return ShuffleNetV2(
        stage_repeats=[4, 8, 4],
        stage_out_channels=[24, 176, 352, 704, 1024],
        in_channels=in_channels,
        num_classes=num_classes
    )


def shufflenetv2_x2_0(in_channels=3, num_classes=200):
    return ShuffleNetV2(
        stage_repeats=[4, 8, 4],
        stage_out_channels=[24, 244, 488, 976, 2048],
        in_channels=in_channels,
        num_classes=num_classes
    )


if __name__ == "__main__":
    model = shufflenetv2_x0_5(num_classes=100)
    print(model)