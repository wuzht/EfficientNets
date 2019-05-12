# EfficientNets
> Implementation of [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

[TOC]

## Introduction

> Version:
>
> * Python 3.7.3
> * pytorch 1.0.1
> * torchvision 0.2.2



## Dataset

The dataset used in the paper ([ShuffleNet V2](https://arxiv.org/abs/1807.11164)) is the ImageNet dataset (ILSVRC2012). However, the size of the full ImageNet is extremely large with a size of over 150 GB. Due to the limitation of GPU resources and time, I have to choose a smaller but similar dataset to conduct experiments.

The dataset chosen is Tiny ImageNet. Tiny ImageNet Challenge is the default course project for Stanford [CS231N](http://cs231n.stanford.edu/). It is similar to the dataset in the full ImageNet ([ILSVRC](http://www.image-net.org/challenges/LSVRC/2012/index)). 

Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images. The training and validation sets are released with images and annotations. All images are 64x64 colored ones.

The training and validation sets of Tiny Imagenet are used here.



## Implementation

### Hyper parameters

Hyper parameters and protocols are exactly the same as [ShuffleNet v1](https://arxiv.org/abs/1707.01083).

* optimizer: SGD
* batch size: 256 on 8 GPUs (32 per GPU)
* weight decay: 4e-5
* momentum: 0.9
* learning rate linear-decay learning rate policy (decreased from 0.5 to 0)



## Results



## Problems



## TODO



## References

* [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
* [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)