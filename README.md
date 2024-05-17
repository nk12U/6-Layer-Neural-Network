## Overview
This is the course material about C Programming Ensyu and code.

C Programming Ensyu is a speciallized subject of Kyoto University Undergraduate School of Electrical and Electronic Engineering.

In this course students need to complete 6 Layer Neural Network, which is able to classify MNIST dataset.

## Training Part
for implementating 3Layer

\$gcc Training_3Layer_Gauss.c -lm -Wall

\$./a.out fc_3Layer.dat

--Training is executed.--

\$ Do you save? Y-0 N-1

if you want to save the weight

type 0

## Learning Part
for implementating 3Layer

\$gcc Inference_3Layer.c -lm -Wall

if you want to inference 0 by the model

\$./a.out fc_3Layer.dat default_0.bmp
