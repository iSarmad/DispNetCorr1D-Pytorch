import tensorflow as tf
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck,Variable,Function


scale = 0.95 ; #make it 0.05 for grad check
gradCheck = False
# Reading the Images
x = imresize(imread('0063L.png'),scale)
y = imresize(imread('0063R.png'),scale)

x = x.astype(float)/255
y = y.astype(float)/255


class Corr(nn.Module):
    def __init__(self,max_disp=10):
        super(Corr, self).__init__()
        self.max_disp = max_disp
    def forward(self,x,y):
        corr_tensors = []
        for i in range(-self.max_disp, 0, 1):
            s1 = y.narrow(3, 0, y.shape[3] + i)
            shifted = F.pad(s1,(-i, 0, 0, 0), "constant", 0.0)
            corr = torch.mean(shifted * x, 1)
            corr_tensors.append(corr)
        for i in range(self.max_disp + 1):
            s2 = x.narrow(3, i, x.shape[3] - i)
            shifted = F.pad(s2,(0, i, 0, 0), "constant", 0.0)
            corr = torch.mean(shifted * y, 1)
            corr_tensors.append(corr)

        temp = torch.stack(corr_tensors)
        out = torch.transpose(temp, 0, 1)
        return torch.mean(out, 1).squeeze()



def correlation_map(x, y, max_disp):
    corr_tensors = []
    for i in range(-max_disp, 0, 1):
        shifted = tf.pad(tf.slice(y, [0]*4, [-1, -1, y.shape[2].value + i, -1]),
                         [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
        corr_tensors.append(corr)
    for i in range(max_disp + 1):
        shifted = tf.pad(tf.slice(x, [0, 0, i, 0], [-1]*4),
                         [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
        corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
        corr_tensors.append(corr)
    return tf.squeeze(tf.reduce_mean(tf.transpose(tf.stack(corr_tensors),
                        perm=[1, 2, 3, 0]),3),0)

# Tensor flow
xf = tf.convert_to_tensor(x)
yf = tf.convert_to_tensor(y)

xf = tf.expand_dims(xf,0)
yf = tf.expand_dims(yf,0)

# Pytorch

x1 = torch.from_numpy(x.transpose(2, 0, 1))
y1 = torch.from_numpy(y.transpose(2, 0, 1))

xt = torch.autograd.Variable(x1.cuda())
yt = torch.autograd.Variable(y1.cuda())

xt = xt.unsqueeze(0)
yt = yt.unsqueeze(0)


disp = 10 # Value for displacement

netf = correlation_map(xf,yf,disp) # Using Tensorflow function

obj1 = Corr() # Using Pytorch Function

""" Grad Check to ensure that my pytorch implementation has correct gradient flow"""
if gradCheck ==True :
    input = (Variable(x1.unsqueeze(0).cuda().double(), requires_grad=True), Variable(y1.unsqueeze(0).cuda().double(), requires_grad=True),)
    input2 = (Variable(torch.randn(1,3,27,30).double(), requires_grad=True),Variable(torch.randn(1,3,27,30).double(), requires_grad=True),)
    test = gradcheck(obj1, input, eps=1e-6, atol=1e-4)
    print(test) # if correct the output should be True


# Using my Pytorch 1D correlation function to obtain output
out = obj1(xt,yt)

out = out.data.cpu().numpy()


with tf.Session() as sess:


    temp = sess.run(netf) # Obtaining output from Tensor flow


    # Plotting
    plt.figure(2)
    ax = plt.subplot(221)
    ax.set_title("Tensorflow Corr1D")
    plt.imshow(temp)
    ax = plt.subplot(222)
    ax.set_title("Pytorch Corr1D")
    plt.imshow(out)
    ax = plt.subplot(223)
    ax.set_title("Left Image")
    plt.imshow(x)
    ax = plt.subplot(224)
    ax.set_title("Right Image")
    plt.imshow(y)
    plt.show(2)



