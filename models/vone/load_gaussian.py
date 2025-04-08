import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
import pickle as pkl
from __init__ import get_model
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import seaborn as sns;
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
print(datetime.now().strftime("%d_%m_%Y"))





def main():
    epath = "/home/wcortez/Documents/models/Epoch_48, trainex128_channels.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(epath, map_location='cuda')
    
    cnum = state['net'][f'module.dn_block.a'].cpu().numpy().shape[0]


    thetaD = state['net'][f'module.dn_block.thetaD']
    a = state['net'][f'module.dn_block.a']
    p = state['net'][f'module.dn_block.p']
    sig = state['net'][f'module.dn_block.sig']

    x = torch.linspace(-12, 12, 12 * 2 + 1, device=device)
    y = torch.linspace(-12, 12, 12 * 2 + 1, device=device)
    xv, yv = torch.meshgrid(x, y)
    xv = xv.expand(cnum, cnum, 12 * 2 + 1, 12 * 2 + 1)
    yv = yv.expand(cnum, cnum, 12 * 2 + 1, 12 * 2 + 1)

    xrot = xv * torch.cos(thetaD) + yv * torch.sin(thetaD)
    yrot = -xv * torch.sin(thetaD) + yv * torch.cos(thetaD)

    gaussian_bank = (abs(a) /
                    (2 * torch.pi * p * sig)) * \
                    torch.exp(-0.5 * ((((xrot) ** 2) / p ** 2) +
                                    (((yrot) ** 2) / sig ** 2)))
    
    selected_gaussian = gaussian_bank[0, 0].cpu().numpy()  # Example: for channels 0,0


    plt.imshow(selected_gaussian, cmap='jet')
    plt.colorbar()
    plt.title("Gaussian Filter for Channels 0,0")
    plt.savefig("gaussian.png")





main()

