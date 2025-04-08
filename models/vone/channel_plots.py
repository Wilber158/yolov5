import torch
from __init__ import get_model
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import torch
from datetime import datetime
from matplotlib.ticker import LinearLocator
from tqdm import tqdm
print(datetime.now().strftime("%d_%m_%Y"))

v1_model = get_model(model_arch=None, pretrained=False, noise_mode=None, div_norm=False, simple_channels=64, complex_channels=64).module
#epoch_num = 32
state = torch.load(f'C:/Users/andre/Desktop/Epoch/Gauss for loop/4_7_2022_128c.pth',
                   map_location='cuda')
state2 = torch.load(f'C:/Users/andre/Desktop/Epoch/Gauss for loop/epoch_9, Div_Norm = 1, VOneNet = 1, Channels = 12, Stride = 2, Color Channels = 1, Noise = None.pth',
                   map_location='cuda')
'''
dn_block_params = ['thetaD', 'p', 'sig', 'a', 'bias', 'gaussian_bank']
for param in dn_block_params:
    print(param + ':')
    print(state['net'][f'module.dn_block.{param}'].shape)
    print(state['net'][f'module.dn_block.{param}'])
'''
'''
for i in range(num_channels):
    v1_k = v1_model.vone_block.simple_conv_q0.weight[i,:,:,:].cpu().numpy().mean(axis=0)
    v1_k = v1_k / np.amax(np.abs(v1_k))/2+0.5
    im_h=ax[i//max_columns, np.mod(i,max_columns)].imshow(v1_k, cmap='gray')
#     ax[i//num_channels, np.mod(i,num_channels)].set_xlim([0, 223])
    im_h.set_clim([0, 1])
    ax[i//max_columns, np.mod(i,max_columns)].set_axis_off()
plt.show()
'''



channels = 128
step = 1

#print(gauss_bank.shape)
Order = np.zeros((1, v1_model.vone_block.orientations.shape[0]))
theta = v1_model.vone_block.orientations.cpu()
Order[0, :] = v1_model.vone_block.orientations.cpu()
#print(Order[0,:])
Order = Order[0,:].argsort()
sigx = v1_model.vone_block.sigx

sigy = v1_model.vone_block.sigy

ratio = np.zeros((v1_model.vone_block.sigy.shape[0]))
for i in range(len(ratio)):
    #if (sigx[i]!= 0 and sigy[i]!=0):
        if sigx[i] > sigy[i]:
            ratio[i] = sigy[i] / sigx[i]
        else:
            ratio[i] = sigx[i] / sigy[i]


ori_channel_idx = []
for i in range(len(ratio)):
    if ratio[i] < 0.75 and Order[i]<128:
    #if ratio[i] < 0.5:
        ori_channel_idx += [i]
ori_channel_count = len(ori_channel_idx)



oriOrder = np.zeros((ori_channel_count))

idx = 0
for i in range(len(Order)):
    if (i in ori_channel_idx):
        oriOrder[idx] = int(Order[i])
        idx+=1
#oriOrder = oriOrder[oriOrder != 0]
title = 'Sigx Sigy Average'
#for i in [state['net'][f'module.dn_block.thetaD'], , , state['net'][f'module.dn_block.a']]:
#amp_bank = state['net'][f'module.dn_block.p']
#print(amp_bank.shape)
#amp_bank = amp_bank[:,:]
amp_bankx = state['net'][f'module.dn_block.sig']
amp_banky = state['net'][f'module.dn_block.p']
amp_bank = (amp_bankx + amp_banky)/2
#amp_bank = (amp_bankx / amp_banky)
sum_amp_bank = np.zeros((channels,channels))
thetaDiff = np.zeros((channels,channels))
p = 0
sum_for_dist = np.zeros((channels*channels,1))
avg_for_dist = np.zeros((channels,1))
#This is the amplitude code
for i in range(channels):
    for j in range(channels):
        sum_amp_bank[i, j] = amp_bank[Order[i], Order[j]]
        thetaDiff[i, j] = theta[Order[i]] - theta[Order[j]]
avg_for_ori_diff = np.zeros((2,channels*channels))
avg_for_ori_diff[0,:] = sum_amp_bank.flatten()
avg_for_ori_diff[1,:] = thetaDiff.flatten()
sum_amp_bank_plot =  sum_amp_bank.flatten()
#for i in range(channels):
    #for j in range(channels):


for i in range(channels):
    avg_for_dist[i] = np.average(sum_amp_bank[:, i])

ori_amp_bank = np.zeros((len(oriOrder), len(oriOrder)))
for i in range(len(oriOrder)):
    for j in range(len(oriOrder)):
        ori_amp_bank[i, j] = amp_bank[int(oriOrder[i]), int(oriOrder[j])]
ori = ori_amp_bank.copy()

ori_amp_bankROW = ori_amp_bank
ori_amp_bankCOLUMN = ori_amp_bank

for i in range(ori_amp_bank.shape[0]):
    ori_amp_bankROW[i, :] = ori_amp_bankROW[i, :] / np.max(ori_amp_bankROW[i, :])

for j in range(ori_amp_bank.shape[0]):
    ori_amp_bankCOLUMN[:,j] = ori_amp_bankCOLUMN[:,j] / np.max(ori_amp_bankCOLUMN[:, j])

ori_amp_bankNORMCOL = (ori_amp_bankROW+ori_amp_bankCOLUMN)/2


#ori_amp_bank = np.clip(ori_amp_bank,0,.2)
#ori_amp_plot = np.sum(ori_amp_bank, axis=0)
#plt.imshow(ori_amp_bank)
#sum_hist = sum_amp_bank.flatten()
#z = np.polyfit(avg_for_ori_diff[1, 1:1000], avg_for_ori_diff[0, 1:1], 10)
#p = np.poly1d(z)
plt.pcolor(ori)
#plt.hist(sum_amp_bank_plot)
#plt.scatter(avg_for_ori_diff[1, :], avg_for_ori_diff[0, :])
#plt.plot(avg_for_ori_diff[1, 1:50], p(avg_for_ori_diff[0, 1:50]), '-r')
plt.title(title)
#plt.xlabel('Difference in Theta')
#plt.ylabel('Amplitude of Normalizing Gaussian')
plt.colorbar()
#
# plt.hist(avg_for_dist)
plt.show()

#for i in range(channels):
    #avg_for_dist[i] = np.average(sum_amp_bank[:,i])
    #amp_bank = amp_bank / np.max(amp_bank)
#sum_for_dist = np.clip(sum_for_dist,0,0.0001)
#sum_for_dist = sum_for_dist/np.max(sum_for_dist)
#plt.hist(avg_for_dist)
#plt.title(f'Distribution of Average (Sigx+Sigy)/2 Per Channel - Variation of {round(np.var(avg_for_dist),4)}')
#plt.show()

#sum_amp_bank = sum_amp_bank / np.max(sum_amp_bank)
#sum_amp_bank = sum_amp_bank**2
inc = 25
#n = 215
#limit = 1
#print(sum_amp_bank)
#sum_amp_bank = np.clip(sum_amp_bank, 0, limit)
#x = np.arange(torch.min(theta), torch.max(theta), torch.max(theta)/channels)
#rows = 8

'''


fig1, ax1 = plt.subplots(rows, 2, figsize=(10, 10))
for i in range(rows):
    channel = i * inc

    #ax.imshow(sum_amp_bank, cmap = 'gray')

    #z = np.polyfit(x, sum_amp_bank[channel, :], 20)
    #p = np.poly1d(z)
    ax1[i,0].bar(x, sum_amp_bank[channel, :], width=0.4)

    #ax1[i,0].plot(x, p(x)*10,'r')
    #ax.plot(np.arange(torch.min(theta), torch.max(theta), torch.max(theta)/channels), sum_amp_bank[:, 1])
    #ax.plot(np.arange(torch.min(theta), torch.max(theta), torch.max(theta)/channels), sum_amp_bank[:, 2])
    #ax.axes.xaxis.set_visible(False)
    ax1[i,0].axes.yaxis.set_visible(False)

    v1_k = v1_model.vone_block.simple_conv_q0.weight[Order[channel], :, :, :].cpu().numpy().mean(axis=0)
    v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
    ax1[i,1].imshow(v1_k, cmap='gray')
    ax1[i,1].axes.yaxis.set_visible(False)
    channels = 40
    step = 10

topknormalizing = 5
channelsinquestion = 6

fig4, ax4 = plt.subplots(channelsinquestion, topknormalizing+1)
for n in range(channelsinquestion):
    for s in range(topknormalizing+1):
        if s==0:
            if n*step<128:
                v1_k = v1_model.vone_block.simple_conv_q0.weight[Order[n * step], :, :, :].cpu().numpy().mean(axis=0)
            elif n*step>=128:
                v1_k = (torch.sqrt(v1_model.vone_block.simple_conv_q0.weight[Order[n * step], :, :, :].cpu().numpy().mean(axis=0)**2 +
                                  v1_model.vone_block.simple_conv_q1.weight[Order[n * step], :, :, :].cpu().numpy().mean(axis=0)**2)/np.sqrt(2))
            v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
            ax4[n,s].imshow(v1_k)
        else:
            amp_order = sum_amp_bank[:,n * step].argsort()
            if n*step<128:
                v1_k = v1_model.vone_block.simple_conv_q0.weight[amp_order[s], :, :, :].cpu().numpy().mean(axis=0)
            elif n*step>=128:
                v1_k = (torch.sqrt(
                    v1_model.vone_block.simple_conv_q0.weight[amp_order[s], :, :, :].cpu().numpy().mean(
                        axis=0) ** 2 +
                    v1_model.vone_block.simple_conv_q1.weight[amp_order[s], :, :, :].cpu().numpy().mean(
                        axis=0) ** 2) / np.sqrt(2))
            v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
            ax4[n,s].imshow(v1_k, cmap='gray')
        ax4[n,s].axes.xaxis.set_visible(False)
        ax4[n,s].axes.yaxis.set_visible(False)
plt.subplots_adjust(wspace=0.1, hspace=0.1)






channels = 25
step = 10
fig2, ax2 = plt.subplots(1, channels, figsize=(10, 10))

for i in range(channels):
    v1_k = v1_model.vone_block.simple_conv_q0.weight[Order[i * step], :, :, :].cpu().numpy().mean(axis=0)
    v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
    ax2[i].imshow(v1_k, cmap='gray')
    ax2[i].axes.xaxis.set_visible(False)
    ax2[i].axes.yaxis.set_visible(False)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()


gauss_bank = state2['net'][f'module.dn_block.gaussian_bank']
thetaOrder = np.zeros((2, v1_model.vone_block.orientations.shape[0]))
thetaOrder[0, :] = v1_model.vone_block.orientations.cpu()
thetaOrder[1, :] = v1_model.vone_block.simple_conv_q0.weight[:, 0, 0, 0].cpu()
thetaOrder = thetaOrder[0].argsort()

channels = 12
step = 1
fig, ax = plt.subplots(channels + 1, channels + 1, figsize=(8, 8))

for x in range(channels):
    for y in range(channels):
        if (y == 0):
            v1_k = v1_model.vone_block.simple_conv_q0.weight[thetaOrder[x*step], :, :, :].cpu().numpy().mean(axis=0)
            v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
            ax[x, y].imshow(v1_k, cmap='gray')
        elif (x == 0):
            v1_k = v1_model.vone_block.simple_conv_q0.weight[thetaOrder[y*step], :, :, :].cpu().numpy().mean(axis=0)
            v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
            ax[x, y].imshow(v1_k, cmap='gray')
        else:
            ax[x, y].imshow(gauss_bank[thetaOrder[x*step], thetaOrder[y*step]].cpu())
        ax[x, y].axes.xaxis.set_visible(False)
        ax[x, y].axes.yaxis.set_visible(False)
fig.delaxes(ax[0, 0])
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

'''
