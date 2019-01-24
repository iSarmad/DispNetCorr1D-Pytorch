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
This is just a sample so I included two pair of images as an example
```
./data-local/bin/prepare_cifar10.sh
```
## Right Image

![alt text](https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/0063R.png)

## Left image

![alt text](https://github.com/iSarmad/DispNetCorr1D-Pytorch/blob/master/0063L.png)

###  Accuracy Achieved on Test Dataset

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
1. Supervised Only without BN : 76.6% 
2. Mean Teacher without BN: 
    a) Student Model : 83.58%
    b) Teacher Model : 86.78%
3. Mean Teacher with BN
    a) Student Model : 84.4%
    b) Teacher Model : 87.07%
4. Mean Teacher + SNTG with BN
    a) Student Model : 84.6%
    b) Teacher Model : 87.28%
5. Hybrid Network
    a) Student Model : 84.18%
    b) Teacher Model : 87.00%
```


## Running the Training 


### Supervised Model Only (4000 labels of Cifar-10)
Go the parameters.py and change the following flags as follows:

1. supervised_mode = True ( To use only 4000 labels for training)
2. lr = 0.15  ( setting the learning rate)
3. BN = False  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py
 Note that my baseline has not Batch Normalization in it. 
### Mean Teacher Only 
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr = 0.2  ( setting the learning rate)
3. BN = False or True  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py

Note that my baseline has not Batch Normalization in it. However I tested mean teacher with both a BN and without BN
 


### Mean Teacher + SNTG Loss 
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr = 0.2  ( setting the learning rate)
3. BN = True  ( for turning batch Normalization on or off)
4. sntg = True ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py

### HybridNet  
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr_hybrid = 0.2  ( setting the learning rate)
3. BN = True  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main_hybrid.py



## Tensorboard Visualization
To Visualize on Tensorboard, use the following command 
```
tensorboard --logdir=”path to ./ckpt”
```
Note that all the checkpoints are in the ./ckpt folder so simply start a tensorboard session to visualize it. Also all the saved checkpoints for student models are also saved there.
```
1. Baseline : 12-03-18:09/convlarge,Adam,200epochs,b256,lr0.15/test
2. Mean teacher without BN :
   12-03-20:12/convlarge,Adam,200epochs,b256,lr0.15/test
   12-03-23:38/convlarge,Adam,200epochs,b256,lr0.2/test
3. Mean Teacher with BN : 12-05-11:55/convlarge,Adam,200epochs,b256,lr0.2/test
4. Hybrid Net : 12-06-10:58/hybridnet,Adam,200epochs,b256,lr0.2/test
5. SNTG + Meant Teacher: 12-07-00:36/convlarge,Adam,200epochs,b256,lr0.2/test
```


## License

This project is licensed under the MIT License. 
For specific helper function used in this repository please see the license agreement of the Repo linked in Acknowledgement section
## Acknowledgments
My implementation has been inspired from the following sources.

* [Mean Teacher](https://github.com/CuriousAI/mean-teacher) : I have mainly followed the Pytorch Version of this Repo
* [SNTG](https://github.com/xinmei9322/SNTG) - I have understood the concept of SNTG and converted Theano Implementation to Pytorch
* [Hybrid Network](https://github.com/dakshitagrawal97/HybridNet) - I have followed this repository to incorporate reconstruction loss in my implementation. 
