import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import matplotx

'''
for epoch_num in tqdm.tqdm(range(0,98)):
    try:
        state = torch.load(f'C:/Users/andre/Desktop/Surround-Suppresion-for-VOneNet/Epochs Resnet DN/epoch_{epoch_num}.pth', map_location='cuda')
        accVecDN.append(state['acc'])
    except Exception:
        pass
    try:
        state = torch.load(
            f'C:/Users/andre/Desktop/Surround-Suppresion-for-VOneNet/Epochs Resnet NoDN/epoch_{epoch_num}.pth',
            map_location='cuda')
        accVecNODN.append(state['acc'])
    except Exception:
        pass
'''
dn256 = torch.load(f'C:/Users/andre/Desktop/Epoch/6_3_2022/dn256.pth', map_location='cuda')
nodn256 = torch.load(f'C:/Users/andre/Desktop/Epoch/6_3_2022/nodn256.pth', map_location='cuda')
nodn128 = torch.load(f'C:/Users/andre/Desktop/Epoch/6_3_2022/nodn128.pth', map_location='cuda')
res18 = torch.load(f'C:/Users/andre/Desktop/Epoch/6_3_2022/res18.pth', map_location='cuda')

dn256_val = dn256['val_acc']
dn256_train = dn256['train_acc']
dn256_lr = dn256['lr']

nodn256_val = nodn256['val_acc']
nodn256_train = nodn256['train_acc']
nodn256_lr = nodn256['lr']

nodn128_val = nodn128['val_acc']
nodn128_train = nodn128['train_acc']
nodn128_lr = nodn128['lr']

res18_val = res18['val_acc']
res18_train = res18['train_acc']
res18_lr = res18['lr']

fig, ax = plt.subplots(1, 3)
ax[0].set_title('Validation Accuracy')
ax[1].set_title('Training Accuracy')
ax[2].set_title('Learning Rate')

ax[0].set_ylim([0,80])
ax[1].set_ylim([0,80])

ax[0].plot(nodn128_val, 'b')
ax[1].plot(nodn128_train, 'b')
ax[2].plot(nodn128_lr, 'b')

ax[0].plot(nodn256_val, 'r')
ax[1].plot(nodn256_train, 'r')
ax[2].plot(nodn256_lr, 'r')

ax[0].plot(dn256_val, 'g')
ax[1].plot(dn256_train, 'g')
ax[2].plot(dn256_lr, 'g')

ax[0].plot(res18_val, 'm')
ax[1].plot(res18_train, 'm')
ax[2].plot(res18_lr, 'm')

fig.legend(['No DN - 128', 'No DN - 256', 'DN - 256', 'Resnet 18'])
plt.show()

'''
xDN = np.arange(1, len(accVec1)+1, 1)
xNODN = np.arange(1, len(accVec2)+1, 1)
labels = [f'{np.max(accVec1)}%', f'{np.max(accVec2)}%']
accVecDN = np.array(accVec2)
accVecNODN = np.array(accVec1)
yDN = np.full((len(accVec1)), np.max(accVec1))
yNODN = np.full((len(accVec2)), np.max(accVec2))
with plt.style.context(matplotx.styles.dufte):
    plt.figure(figsize=(15,10))
    plt.plot(xDN, accVec1, label=labels[0])
    plt.plot(xNODN, accVec2, label=labels[1])
    plt.legend(['VOne Resnet 18','Base Resnet 18'], loc='lower right')
    plt.plot(xDN, yDN, linestyle='dashed')
    plt.plot(xNODN, yNODN, linestyle='dashed')
    plt.title(f'Accuracy of VOnenet Resnet 18 vs Base Resnet')
    plt.xlabel('Epoch')
    plt.ylabel('% Accuracy')
    matplotx.line_labels()
    plt.show()

lrVec1 = state1['lr']
lrVec2 = state2['lr']

xDN = np.arange(1, len(lrVec1)+1, 1)
xNODN = np.arange(1, len(lrVec2)+1, 1)
labels = [f'{np.min(lrVec1)}', f'{np.min(lrVec2)}']
with plt.style.context(matplotx.styles.dufte):
    plt.figure(figsize=(15,10))
    plt.plot(xDN, lrVec1, label=labels[0])
    plt.plot(xNODN, lrVec2, label=labels[1])
    plt.legend(['VOne Resnet 18','Base Resnet 18'], loc='lower right')
    plt.title(f'Learning Rate of VOnenet Resnet 18 vs Base Resnet')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    matplotx.line_labels()
    plt.show()
'''