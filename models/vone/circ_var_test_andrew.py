#Edited circular Variance Code, mainly changed 64 to 32, used for debugging
#Edit: I no longer think we should change N, so I do not use this version
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from vone.utils import gabor_kernel
from __init__ import get_model
import matplotlib.pyplot as plt
from scipy.stats import circvar

def circ_var(p, fmin=0.3, fmax=0.7, plot=False):
    """
    Returns mean resultant vector in relvant frequency band.
    The length of the resultant vector [np.abs(v)] indicates the degree
    of orientation tuning while its angle [np.angle(v)] indicates the
    preferred orientation.
    #The input must be the sample weights
    """
    p = np.fft.fftshift(p)
    k = p.shape[-1]
    x = np.linspace(-1, 1, k, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx ** 2 + yy ** 2)
#     print(r.shape)
    phi = np.arctan2(xx, yy)
    p_band = p * ((r > fmin) & (r < fmax))
    p_band = p_band / (p_band ** 2).mean()
#     plt.imshow(p_band)
#     plt.show()
    if plot:
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(p, cmap="gray")
        axes[1].imshow(p_band, cmap="gray")
    resultant_vector = (p_band * np.exp(2j * phi))
    print(resultant_vector)
    return resultant_vector


def gausswin(w, d=2):
    k = w.shape[-1]
    ii = np.linspace(-d, d, k, endpoint=True)
    ii, jj = np.meshgrid(ii, ii)
    return np.exp(-(ii ** 2 + jj ** 2) / 2.0)


def angles_circ_var(features, threshold=0.2):
    """Compute filter orientation with circular variance.
    Args:
        features: np.array of shape (feature, size, size) of a batch of features
    Returns:
        np.array of orientation angles.
    """

    #w = np.squeeze(np.array(features))
    w = features
    w = w / (w ** 2).sum(axis=(1, 2), keepdims=True)

    # Power spectrum (windowed)
    N = 64
    #ww = w * gausswin(w)
    #ww = ww / (ww ** 2).sum(axis=(1, 2), keepdims=True)
    #P = np.abs(np.fft.fft2(w, s=(N, N)))
    #ww = features
    #ww = ww / (ww ** 2).sum(axis=(1, 2), keepdims=True)
    P = (np.fft.fft2(w, s=(N, N)))
    #P = abs(P)
    #plt.imshow(P[1])
    #plt.show()
    #print(np.max(P[1,:,:]))
    #print(np.min(P[1,:,:]))
    #P = (np.fft.fft2(w, s=(N, N), norm='forward'))
    #P = P / (P ** 2).mean()
    P = np.fft.fftshift(P)
    p=P
    k = p.shape[-1]
    x = np.linspace(-1, 1, k, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx ** 2 + yy ** 2)
    phi = np.arctan2(xx, yy)
    p_band = p * ((r > 0.01) & (r < 0.99))
    #p_band = abs(p_band * np.exp(2j * phi))
    p_band = p_band * np.exp(2j * phi)
    #RF_Circ_Var = np.array(abs(p_band.sum(axis = (1, 2))/p.sum(axis = (1, 2))))
    #p_band = p_band / (p_band ** 2).mean()
    RF_Circ_Var =1- np.array([circvar(c) for c in abs(p_band)])
    #RF_Circ_Var = abs(np.log(RF_Circ_Var))
    #RF_Circ_Var = RF_Circ_Var/max(RF_Circ_Var)
    return RF_Circ_Var, abs(p_band), abs(P)

v1_model = get_model(model_arch=None, pretrained=False, noise_mode=None, div_norm=False, simple_channels=64, complex_channels=64
                     , map_location='cpu').module
RF = v1_model.vone_block.simple_conv_q0.weight[:,0] #[128 x 31 x 31]
t = .00008 #threshold value,
oriori, p, f = angles_circ_var(RF, t)
print(oriori.shape)
print(max(abs(oriori)))
print(min(abs(oriori)))
f = np.abs(f)
order = np.argsort(oriori)
#oriidx = np.where(~np.isnan(oriori))
#unoriidx = np.where(np.isnan(oriori))
fig, ax = plt.subplots(3, 3)
idx = 0
for i in range(3):
    for j in range(3):
        #plt.title(f"Circular Variance {oriori[order[i]]}")
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
        ax[i, j].imshow((v1_model.vone_block.simple_conv_q0.weight[order[idx], 0, :, :]),  cmap="gray")
        ax[i, j].set_title(round(oriori[order[idx]], 4))
        idx += 1

plt.figure(1)
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
idx = 0
for i in range(3):
    for j in range(3):
        #plt.title(f"Circular Variance {oriori[order[i]]}")
        #ax[i, j].get_xaxis().set_visible(False)
        #ax[i, j].get_yaxis().set_visible(False)
        ax[i, j].imshow((p[order[idx]]),  cmap="gray")
        ax[i, j].set_title(round(oriori[order[idx]], 3))
        idx += 1
plt.figure(2)
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
idx = 0
for i in range(3):
    for j in range(3):
        #plt.title(f"Circular Variance {oriori[order[i]]}")
        #ax[i, j].get_xaxis().set_visible(False)
        #ax[i, j].get_yaxis().set_visible(False)
        ax[i, j].imshow((f[order[idx]]),  cmap="gray")
        ax[i, j].set_title(round(oriori[order[idx]], 3))
        idx += 1

plt.show()
