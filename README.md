# 6-Layer-Neural-Network

## Overview

6 Layer Fully Connected Neural Network classifies MNIST dataset.  

## Network Architecture

<img src = "https://github.com/nk12U/6-Layer-Neural-Network/blob/main/Neural Network Architecture.png">

## Environment

- [WSL2](https://learn.microsoft.com/ja-jp/windows/wsl/install)
- [MinGW-w64](https://www.javadrive.jp/cstart/install/index6.html)

## Command

```
$ git clone https://github.com/nk12U/6-Layer-Neural-Network.git
```

### Training

To implement 6 Layer Network

```
$ gcc Training_6Layer_Gauss.c -lm -Wall

$ ./a.out fc1_6Layer.dat fc2_6Layer.dat fc3_6Layer.dat
```

--Training is executing.--

```
$ Do you save? Y-0 N-1

If you want to save the weight and bias.

$ 0
```

### Inference

To implement 6 Layer Network

```
$ gcc Inference_6Layer.c -lm -Wall

If you want to inference character 0 by the trained model.

$ ./a.out fc1_6Layer.dat fc2_6Layer.dat fc3_6Layer.dat default_0.bmp
```

## Future Work

- Add Adam and Momentum optimization 
- Expand this code to Convolutional Neural Network and CIFAR-10 dataset
- CUDA Implementation
