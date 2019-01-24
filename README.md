# DispNetCorr1D-Pytorch
This repository only provides the 1D correlation function needed for DispNetCorr1D in Pytorch. I mainly made this because there are ample resources for Tensorflow implementation of the following Papers but the Pyotrch implementation have no 1D correlation layer. So I wrote my own. 

* [DispNetCorr1D](https://arxiv.org/abs/1703.01780) A Large Dataset of 
* [FlowNet](https://arxiv.org/pdf/1711.00258.pdf)   Smooth Neighbors on Teacher Graphs for Semi-supervised Learning

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

###  Tensorflow Version

I replicated the tensor flow version from the following repository.

* [dispflownet-tf](https://github.com/fedor-chervinskii/dispflownet-tf) Tensorflow implementation of https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16

### Pytorch Implementation 

To make sure my version is exactly equal to the Tensorflow version I had to make sure that both of the following condition are checked:

1. Forward function
2. Backward function

## Forward Function 

![alt text](https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/sample.png)

## Backward Function 


## License

This project is licensed under the MIT License. 
For specific helper function used in this repository please see the license agreement of the Repo linked in Acknowledgement section
## Acknowledgments
My implementation has been inspired from the following sources.

* [Mean Teacher](https://github.com/CuriousAI/mean-teacher) : I have mainly followed the Pytorch Version of this Repo
* [SNTG](https://github.com/xinmei9322/SNTG) - I have understood the concept of SNTG and converted Theano Implementation to Pytorch
* [Hybrid Network](https://github.com/dakshitagrawal97/HybridNet) - I have followed this repository to incorporate reconstruction loss in my implementation. 
