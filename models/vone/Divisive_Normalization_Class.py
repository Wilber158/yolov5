#Lawrence was here 11/23/21 .....
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from modules import VOneBlock
import torch.nn as nn
import scipy
from scipy import signal
import tensorflow as tf

#Andrew is cute
#Testing Testing ANDREW 10:59
#11/23/202
#surrounding suppressions
#Idea: Do learning with smaller number of channels. Takes WAY too long otherwise

class Identity(nn.Module):
    def forward(self, x):
        return x

class DivisiveNormBlock(nn.Module):

    def __init__(self, channel_num = 512, size = 56, ksize = 3):
        super(DivisiveNormBlock, self).__init__()

        self.channel_num = channel_num
        self.size = size
        self.ksize = ksize

        scale = 90 #Random scale factor I've been playing with.
        self.theta = torch.nn.Parameter(scale * torch.abs(torch.randn(self.channel_num, self.channel_num, device="cuda", requires_grad=True)))# 512 thetas for a channel, 512 channels, same goes for...
        self.p = torch.nn.Parameter(scale * torch.abs(torch.randn(self.channel_num, self.channel_num, device="cuda", requires_grad=True)) + 10)
        self.sig = torch.nn.Parameter(scale * torch.abs(torch.randn(self.channel_num, self.channel_num, device="cuda", requires_grad=True)) + 10)
        self.a = torch.nn.Parameter(scale * torch.abs(torch.randn(self.channel_num, self.channel_num, device="cuda", requires_grad=True)))
        self.nI = torch.nn.Parameter(torch.abs(torch.randn(self.channel_num, self.channel_num, device="cuda", requires_grad=True)))
        self.nU = torch.nn.Parameter(torch.abs(torch.randn(self.channel_num, device="cuda", requires_grad=True)))
        self.bias = torch.nn.Parameter(torch.abs(torch.randn(self.channel_num, device="cuda", requires_grad=True)))
        self.gaussian_bank = torch.zeros(self.channel_num, self.channel_num, self.ksize*2, self.ksize*2, device="cuda")
        self.x = torch.linspace(-self.ksize, self.ksize, self.ksize * 2, device="cuda")
        self.y = torch.linspace(-self.ksize, self.ksize, self.ksize * 2, device="cuda")
        self.xv, self.yv = torch.meshgrid(self.x, self.y)
        self.output = Identity()

        for i in range(self.channel_num):
            for u in range(self.channel_num):
                self.gaussian_bank[i, u, :, :] = self.get_gaussian(i, u)



    def forward(self, x):

        x = self.dn_f(x)

        x = self.output(x)

        return x

    def get_gaussian(self, cc, oc):  #
        xrot = self.xv * torch.cos(self.theta[cc, oc]) + self.yv * torch.sin(self.theta[cc, oc])
        yrot = -self.xv * torch.sin(self.theta[cc, oc]) + self.yv * torch.cos(self.theta[cc, oc])
        g_kernel = (self.a[cc, oc] / \
                   (2 * torch.pi * self.p[cc, oc] * self.sig[cc, oc])) * \
                   torch.exp(-0.5 * ((((xrot)**2 ) /self.p[cc, oc ]**2) + ((yrot)**2 ) /self.sig[cc, oc]**2))

        return self.output(g_kernel)

    def dn_f(self, x):

        batch_size = x.shape[0]
        under_sum = torch.zeros((self.channel_num, self.size, self.size), device="cuda")
        normalized_channels = torch.zeros((batch_size, self.channel_num, self.size, self.size), device="cuda")
        for b in range(batch_size):
            for i in tqdm(range(self.channel_num)):
                for u in range(self.channel_num):
                    under_sum[u] = self.conv_gauss(torch.pow(x[b, i], self.nI[i, u]), self.gaussian_bank[i, u])
                normalized_channels[b, i] = torch.pow(x[b, i], self.nU[i]) / (
                            torch.pow(self.bias[i], self.nU[i]) + torch.sum(under_sum, 0))
        return self.output(normalized_channels)

    def conv_gauss(self, x, gf): #ccf = current channel filter, oc = other channel

        x = torch.reshape(x, (1, 1, self.size, self.size))
        gf = torch.reshape(gf, (1, 1, self.ksize*2, self.ksize*2))
        #56 + 2*(padding) - dilation * (ksize*2 - 1) - 1 = 58 - 9 - 1 = 48
        conv = nn.Conv2d(self.channel_num, self.channel_num, kernel_size=(self.ksize*2, self.ksize*2), bias=False, padding=(self.ksize), device="cuda")
        gauss_filter = torch.nn.Parameter(gf)
        conv.weight = gauss_filter
        output = conv(x)
        output = torch.reshape(output, (output.shape[2], output.shape[3]))
        output = output[1:output.size(0), 1:output.size(1)] #Convolution size is all messy, fix??
        return self.output(output)












#Testing
'''
ex_pass = torch.rand((1, 10, 56, 56))
DNB = DivisiveNormBlock(channel_num=10, size=56, ksize=5) #I think ksize needs to be off for mult_gauss
ex_output = DNB(ex_pass)

v0_k = ex_pass[0, 1, :, :].detach().numpy()#.mean(axis=0)  # Makes it plottable
v0_l = ex_output[0, 1, :, :].detach().numpy()#.mean(axis=0)

v0_k = v0_k / np.amax(np.abs(v0_k)) / 2 + 0.5
v0_l = v0_l / np.amax(np.abs(v0_l)) / 2 + 0.5

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(7, 7)  # Figure size
im_h0 = ax1.imshow(v0_k)
im_h1 = ax2.imshow(v0_l)
plt.show()
'''

#Previous conv_gauss

'''
#ccf is of size ksize*2 x ksize*2. This is the gaussian filter of cc for oc
#oc is of size 56 x 56. We move ccf over oc
#return a 56x56
#current_sum = scipy.signal.convolve2d(oc, ccf, mode='same', boundary='fill', fillvalue=0)
conv = nn.Conv2d(in_channels=self.channel_num, out_channels=self.channel_num, kernel_size=self.ksize, stride=1)
#current_sum = nn.Conv2d(x, gb, kernel_size=self.ksize, stride=1, padding='same') #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
#https://stackoverflow.com/questions/42883547/intuitive-understanding-of-1d-2d-and-3d-convolutions-in-convolutional-neural-n
#current_sum = torch.tensor(current_sum)
current_sum = conv(x, gb)

for x in range(current_sum.shape[0]):
    for y in range(current_sum.shape[1]):
        current_sum[x, y] = torch.sum(padded_oc[x:x+self.ksize*2, y:y+self.ksize*2] * ccf)


return current_sum
'''

#Previous under_sum

'''
for b in range(batch_size):
    print('Input Channel %d' % (b + 1))
    for i in tqdm(range(self.channel_num)): #Current channel

        under_sum = torch.zeros((self.channel_num, self.size, self.size))
        for u in range(self.channel_num): #Other channels
            under_sum[u] = self.conv_gauss(self.gaussian_bank[i, u], x[b, u])
            under_sum[u] = self.remove_and_normalize(under_sum[u], 1, 0)

        under_sum = self.remove_and_normalize(under_sum, 1, 0)
        normalized_channels[b, i] = x[b, i] / torch.nansum(under_sum, 0)
        normalized_channels[b, i] = self.remove_and_normalize(normalized_channels[b, i], 0, 0)
'''

#Show gaussian

'''    
#To plot the gaussians
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111, projection='3d')
surf = ax1.plot_surface(self.xv,self.yv,self.gaussian_bank[i,u])
#im_h0 = ax1.imshow(self.gaussian_bank[i, u])
plt.title("Theta %f, p %f, sig %f, a %f" %(self.theta[i,u], self.p[i,u], self.sig[i,u], self.a[i,u]))
plt.show()
'''