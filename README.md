## Overview

This is the course material about Exercise of Computer Programming in Electrical and Electronic Engineering.

Exercise of Computer Programming in Electrical and Electronic Engineering is a speciallized subject of Kyoto University Undergraduate School of Electrical and Electronic Engineering.

In this course, as the final assignment students need to complete 6 Layer Neural Network, which is able to classify MNIST dataset.

I just confirmed that this code was working on WSL2 and MinGW-w64, so this code may not work in Mac.

## Training

for implementating 6Layer

```
$ gcc Training_6Layer_Gauss.c -lm -Wall

$ ./a.out fc1_6Layer.dat fc2_6Layer.dat fc3_6Layer.dat
```

--Training is executing.--

```
Do you save? Y-0 N-1

if you want to save the weight

type 0
```

## Inference

for implementating 6Layer

```
$ gcc Inference_6Layer.c -lm -Wall

if you want to inference 0 by the trained model

$ ./a.out fc1_6Layer.dat fc2_6Layer.dat fc3_6Layer.dat default_0.bmp
```
