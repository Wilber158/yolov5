import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.vone.utils import gabor_kernel
from torch.distributions import uniform


from models.vone import params
torch.autograd.set_detect_anomaly(True)


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2


        # Initialize parameters
        self.register_buffer('thetaA', torch.zeros(out_channels))
        self.register_buffer('ratioA', torch.zeros(out_channels))
        self.register_buffer('sfA', torch.zeros(out_channels))
        self.register_buffer('sigxA', torch.zeros(out_channels))
        self.register_buffer('sigyA', torch.zeros(out_channels))
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)


    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)


    def initialize(self, sf, theta, sigx, sigy, phase, color_channels):
        device = self.weight.device  # Get the device of self.weight

        # Convert inputs to tensors on the correct device
        sf = torch.tensor(sf, device=device)
        theta = torch.tensor(theta, device=device)
        sigx = torch.tensor(sigx, device=device)
        sigy = torch.tensor(sigy, device=device)
        phase = torch.tensor(phase, device=device)

        for i in range(self.out_channels):
            # Generate Gabor kernel on the correct device
            gabor = gabor_kernel(
                frequency=sf[i].item(),
                sigma_x=sigx[i].item(),
                sigma_y=sigy[i].item(),
                theta=theta[i].item(),
                offset=phase[i].item(),
                ks=self.kernel_size,
                device=device
            )

            # Assign Gabor kernel to self.weight
            with torch.no_grad():
                self.weight[i, :, :, :] = gabor

            # Assign to buffers directly
            self.sfA[i] = sf[i]
            self.thetaA[i] = theta[i]
            self.sigxA[i] = sigx[i]
            self.sigyA[i] = sigy[i]
            self.ratioA[i] = sigx[i] / sigy[i]


class VOneBlock(nn.Module):
    def __init__(self, simple_channels=32, complex_channels=32, stride=4, ksize=25, noise_mode='neuronal',
                 noise_scale=0.35, noise_level=0.07, k_exc=25, color_channels=3):
        super(VOneBlock, self).__init__()
        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.stride = stride
        self.ksize = ksize
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level
        self.k_exc = k_exc
        self.color_channels = color_channels
        self.out_channels = simple_channels + complex_channels

        self.color_channels = color_channels

        self.input_size = None

        # Create GFB
        self.simple_conv_q0 = GFB(self.color_channels, self.out_channels, self.ksize, self.stride)
        self.simple_conv_q1 = GFB(self.color_channels, self.out_channels, self.ksize, self.stride)


        # Other layers
        self.simple = nn.ReLU(inplace=False)
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = nn.ReLU(inplace=False)
        self.output = Identity()


        # Initialize parameters as None; they'll be set in the initialize method
        self.sf = None
        self.theta = None
        self.sigx = None
        self.sigy = None
        self.phase = None
        self.fixed_noise = None


    def initialize(self, x):
        # Calculate parameters based on image size
        image_size = x.shape[-1]
        self.input_size = image_size
        visual_degrees = 8
        ppd = image_size / visual_degrees


        # Generate Gabor parameters
        sf, theta, phase, nx, ny = params.generate_gabor_param(
            self.simple_channels + self.complex_channels, 0, False, 0.75, 11.3, 0
        )


        sf /= ppd
        sigx = nx / sf
        sigy = ny / sf
        theta = theta / 180 * np.pi
        phase = phase / 180 * np.pi


        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase


        # Initialize GFB modules
        self.simple_conv_q0.initialize(
            sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy, phase=self.phase, color_channels=self.color_channels
        )
        self.simple_conv_q1.initialize(
            sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy, phase=self.phase + np.pi / 2, color_channels=self.color_channels
        )


        # Assign parameters from GFB modules
        self.thetaA = self.simple_conv_q0.thetaA
        self.ratioA = self.simple_conv_q0.ratioA
        self.sfA = self.simple_conv_q0.sfA
        self.sigxA = self.simple_conv_q0.sigxA
        self.sigyA = self.simple_conv_q0.sigyA


    def forward(self, x):
        if self.sf is None or x.shape[-1] != self.input_size:
            self.initialize(x)

        with torch.no_grad():
            # Gabor activations
            x = self.gabors_f(x)

            self.fix_noise(x)


            # Apply noise
            x = self.noise_f(x)


            # Output
            x = self.output(x)

            print(f'VOneBlock Output - Mean: {x.mean().item()}, Std: {x.std().item()}, Max: {x.max().item()}, Min: {x.min().item()}')
    
            return x


    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        epsilon = 1e-6  # Small constant to prevent sqrt(0)
        sum_sq = s_q0[:, self.simple_channels:, :, :] ** 2 + s_q1[:, self.simple_channels:, :, :] ** 2
        c_input = torch.sqrt(sum_sq + epsilon) / np.sqrt(2)
        c = self.complex(c_input)
        s_input = s_q0[:, :self.simple_channels, :, :].clone()
        s = self.simple(s_input)
        return self.gabors(self.k_exc * torch.cat((s, c), 1))


    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 1e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                print()
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
            else:
                x += torch.randn_like(x) * torch.sqrt(F.relu(x.clone()) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        elif self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.randn_like(x) * self.noise_scale
        return self.noise(x)


    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level


    def fix_noise(self, x):
        if self.noise_mode:
            self.fixed_noise = torch.randn_like(x)


    def unfix_noise(self):
        self.fixed_noise = None


# Divisive Normalization Block
class DivisiveNormBlock(nn.Module):
    def __init__(self, channel_num=512, ksizeDN=12, use_full_image_net=0, restore_path=None, map_location=None):
        super().__init__()
        self.channel_num = channel_num
        self.ksizeDN = ksizeDN
        self.use_full_image_net = use_full_image_net

        if use_full_image_net == 1:
            # Load pre-trained parameters
            state = torch.load(restore_path, map_location=map_location)
            new_state_dict = {k.replace('module.', ''): v for k, v in state['net'].items()}
            self.thetaD = nn.Parameter(new_state_dict['dn_block.thetaD'], requires_grad=False)
            self.p = nn.Parameter(new_state_dict['dn_block.p'], requires_grad=False)
            self.sig = nn.Parameter(new_state_dict['dn_block.sig'], requires_grad=False)
            self.a = nn.Parameter(new_state_dict['dn_block.a'], requires_grad=False)
        else:
            # Initialize parameters
            grad = True
            self.thetaD = nn.Parameter(uniform.Uniform(0, np.pi).sample([channel_num, channel_num, 1, 1]), requires_grad=grad)
            self.p = nn.Parameter(uniform.Uniform(2, 6).sample([channel_num, channel_num, 1, 1]), requires_grad=grad)
            self.sig = nn.Parameter(uniform.Uniform(2, 6).sample([channel_num, channel_num, 1, 1]), requires_grad=grad)
            self.a = nn.Parameter(torch.abs(torch.randn(channel_num, channel_num, 1, 1)), requires_grad=grad)

    def forward(self, x):
        x = self.dn_f(x)
        return x

    def dn_f(self, x):
        device = x.device
        batch_size, _, height, width = x.size()

        # Create meshgrid for Gaussian kernel based on ksizeDN
        ksize = self.ksizeDN * 2 + 1
        x_range = torch.linspace(-self.ksizeDN, self.ksizeDN, ksize, device=device)
        y_range = torch.linspace(-self.ksizeDN, self.ksizeDN, ksize, device=device)

        xv = x_range.view(1, 1, 1, ksize).expand(self.channel_num, self.channel_num, ksize, ksize)
        yv = y_range.view(1, 1, ksize, 1).expand(self.channel_num, self.channel_num, ksize, ksize)

        thetaD = self.thetaD.to(device)
        p = self.p.to(device)
        sig = self.sig.to(device)
        a = self.a.to(device)

        # Create Gaussian bank using rotated coordinates
        xrot = xv * torch.cos(thetaD) + yv * torch.sin(thetaD)
        yrot = -xv * torch.sin(thetaD) + yv * torch.cos(thetaD)
        pi = torch.acos(torch.zeros(1, device=device)).item() * 2  # pi value
        gaussian_bank = (torch.abs(a) / (2 * pi * p * sig)) * \
                        torch.exp(-0.5 * ((xrot ** 2 / p ** 2) + (yrot ** 2 / sig ** 2)))
        gaussian_bank = F.relu(gaussian_bank)

        # Normalize input
        bias = torch.ones(1, self.channel_num, 1, 1, device=device)
        padding_size = self.ksizeDN

        with torch.no_grad():
            under_sum = F.conv2d(x, gaussian_bank, stride=1, padding=padding_size)
            epsilon = 1e-6
            x = x / (bias + under_sum + epsilon)

        return x




