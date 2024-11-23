# 6-Layer-Neural-Network

## Overview

This repository is 6 Layer Neural Network written in C, which classify MNIST dataset.  
This is the final assignment of Exercise of Computer Programming in Electrical and Electronic Engineering at Kyoto University.

## Neural Network Architecture

<img src = "https://github.com/nk12U/6-Layer-Neural-Network/blob/main/Neural Network Architecture.png">

## Environment

- [MinGW-w64](https://www.javadrive.jp/cstart/install/index6.html)
- [WSL2](https://learn.microsoft.com/ja-jp/windows/wsl/install)

```
$ git clone https://github.com/nk12U/6-Layer-Neural-Network
```

## Training

To implement 6Layer

```
$ gcc Training_6Layer_Gauss.c -lm -Wall

$ ./a.out fc1_6Layer.dat fc2_6Layer.dat fc3_6Layer.dat
```

--Training is executing.--

```
$ Do you save? Y-0 N-1

if you want to save the weight and bias

$ 0
```

## Inference

To implement 6Layer

```
$ gcc Inference_6Layer.c -lm -Wall

if you want to inference 0 by the trained model

$ ./a.out fc1_6Layer.dat fc2_6Layer.dat fc3_6Layer.dat default_0.bmp
```

## Future work

- Expand this code to Convolutional Neural Network and CIFAR-10 dataset.
- CUDA Implementation
