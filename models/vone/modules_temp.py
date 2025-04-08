
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vone.utils import gabor_kernel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        kernel_size = 5
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (2, 2)
        self.padding = (kernel_size // 2, kernel_size // 2)
        #self.padding = (2,6)
        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        #Generate 512 random ints between 0 and 2.
        #Each channel has the same weights?????
        #Within each channel each neuron have the same weights?
        #Gabor function?
        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for i in range(self.out_channels):
            #Gives the weights for the same unit per each channel
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                                             theta=theta[i], offset=phase[i], ks=self.kernel_size[0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=5, stride=2, input_size=64, ksizeDN=1, size=32):
        super().__init__()

        self.in_channels = 1 #1 for greyscale

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=True)
        self.output = Identity()

        self.channel_num = self.out_channels
        self.size = size
        self.ksizeDN = ksizeDN

        self.thetaD = torch.nn.Parameter(torch.abs(torch.randn(self.channel_num, self.channel_num,
                                                                      requires_grad=True)))
        self.p = torch.nn.Parameter(
            torch.abs(torch.randn(self.channel_num, self.channel_num, requires_grad=True)))
        self.sig = torch.nn.Parameter(
            torch.abs(torch.randn(self.channel_num, self.channel_num, requires_grad=True)))
        self.a = torch.nn.Parameter(
            torch.abs(torch.randn(self.channel_num, self.channel_num, requires_grad=True)))
        self.nU = torch.nn.Parameter(torch.abs(torch.randn(self.channel_num, requires_grad=True)))
        #self.bias = torch.nn.Parameter(torch.abs(torch.randn(self.channel_num, requires_grad=True)))
        self.gaussian_bank = torch.nn.Parameter(torch.zeros(self.channel_num, self.channel_num, self.ksizeDN * 2+ 1,
                                         self.ksizeDN * 2+ 1), requires_grad=False)
        self.x = torch.nn.Parameter(torch.linspace(-self.ksizeDN, self.ksizeDN, self.ksizeDN * 2 + 1),
                                    requires_grad=False)
        self.y = torch.nn.Parameter(torch.linspace(-self.ksizeDN, self.ksizeDN, self.ksizeDN * 2 + 1),
                                    requires_grad=False)
        xvT, yvT = torch.meshgrid(self.x, self.y)
        self.xv = torch.nn.Parameter(xvT.clone().detach().requires_grad_(False))
        self.yv = torch.nn.Parameter(yvT.clone().detach().requires_grad_(False))

        for i in range(self.channel_num):
            for u in range(self.channel_num):
                self.gaussian_bank[i, u, :, :] = self.get_gaussian(i, u)

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]

        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)

        x = self.dn_f(x)

        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :, :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))

    def noise_f(self, x):

        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels, int(self.input_size/self.stride),
                                 int(self.input_size/self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean, scale=1).rsample().to(device)

    def unfix_noise(self):
        self.fixed_noise = None

    def dn_f(self, x):

        batch_size = x.shape[0]
       # under_sum = torch.zeros((1, self.channel_num, self.size, self.size), device=x.device)
        normalized_channels = torch.zeros((batch_size, self.channel_num, self.size, self.size), device=x.device)
        for b in tqdm(range(batch_size)):
            #for i in range(self.channel_num):
                #for u in range(self.channel_num):
                    #under_sum[u] = self.conv_gauss(torch.pow(x[b, u], self.nU[u]), self.gaussian_bank[i, u])
                #normalized_channels[b, i] = torch.pow(x[b, i], self.nU[i]) / (
                     #torch.pow(self.bias[i], self.nU[i]) + torch.sum(under_sum, 0))
            under_sum = self.conv_gauss(x[b], self.gaussian_bank)
            normalized_channels[b] = x[b] / under_sum #(torch.pow(self.bias, self.nU) + under_sum)
        return normalized_channels

    def conv_gauss(self, x_conv, gauss_conv):
        x_conv = torch.reshape(x_conv, (1, self.channel_num, self.size, self.size))
        #gauss_conv = torch.reshape(gauss_conv, (1, 1, self.ksizeDN * 2+ 1, self.ksizeDN * 2+ 1))
        p = int((self.ksizeDN*2)/2)
        output = F.conv2d(x_conv, gauss_conv, stride=1, padding = 1)
       # output = torch.reshape(output, (self.size, self.size))
        return output

    def get_gaussian(self, cc, oc):
        xrot = (self.xv * torch.cos(self.thetaD[cc, oc]) + self.yv * torch.sin(self.thetaD[cc, oc]))
        yrot = (-self.xv * torch.sin(self.thetaD[cc, oc]) + self.yv * torch.cos(self.thetaD[cc, oc]))
        g_kernel = torch.tensor((self.a[cc, oc] /
                    (2 * torch.pi * self.p[cc, oc] * self.sig[cc, oc])) * \
                   torch.exp(-0.5 * ((((xrot) ** 2) / self.p[cc, oc] ** 2) +
                                     ((yrot) ** 2) / self.sig[cc, oc] ** 2)))

        return g_kernel








