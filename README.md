# DispNetCorr1D-Pytorch
This repository only provides the 1D correlation function needed for DispNetCorr1D in Pytorch. I mainly made this because there are ample resources for Tensorflow implementation of the following papers but the Pytorch implementation have no 1D correlation layer. So I wrote my own. 

* [DispNetCorr1D](https://arxiv.org/abs/1703.01780) A Large Dataset to Train Convolutional Networks
for Disparity, Optical Flow, and Scene Flow Estimation 
* [FlowNet](https://arxiv.org/pdf/1711.00258.pdf)   FlowNet: Learning Optical Flow with Convolutional Networks

I only used Pytorch functions so a faster version can be implemented in CUDA. This version was also fast enough for training in my experiments.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

First Install all the relevant requirements given in the requirements.txt file in your pip or anaconda environments. 

 
### Dataset 
This repo is just a sample so I included one pair of left and right images as an example as shown below

## Right Image

![alt text](https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/0063R.png)

## Left image

![alt text](https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/0063L.png)

### Running the Code

Execute the main.py file.

###  Tensorflow Version

I replicated the tensor flow version from the following repository.

* [dispflownet-tf](https://github.com/fedor-chervinskii/dispflownet-tf) Tensorflow implementation of https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16

The mentioned repository provides the CUDA version also which is not covered in this repository. 

### Pytorch Implementation 

To make sure my version is exactly equal to the Tensorflow version I had to make sure that both of the following condition are checked:

1. Forward function
2. Backward function

## Forward Function 
As you can confirm by running the program that the Pytorch Version is exactly similar to TF version  as shown below:
![alt text](https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/sample.png)

## Backward Function 
I made sure using Grad check that gradients were flowing correctly. For this :

1. Set gradCheck = True
2. Set Scale = 0.05 

## License

This project is licensed under the MIT License. 
For specific helper function used in this repository please see the license agreement of the Repo linked in Acknowledgement section

## Acknowledgments and Further References 
My implementation has been inspired from the following sources.

** [dispflownet-tf](https://github.com/fedor-chervinskii/dispflownet-tf) Tensorflow implementation of https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16

You can use my function with the following repository: 

** [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) Pytorch implementation of FlowNet by Dosovitskiy et al.

** [SfmLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) Pytorch version of SfmLearner from Tinghui Zhou et al.

In the above simply modify FlowNet or DispnetS and use my 1D Correlation function to get DispNetCorr1D

