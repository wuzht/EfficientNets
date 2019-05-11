# EfficientNets
> Implementation of [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

[TOC]

## Introduction



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