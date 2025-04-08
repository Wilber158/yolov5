
from collections import OrderedDict
from torch import nn
import torch
from modules import VOneBlock, DivisiveNormBlock
from back_ends import Bottleneck, AlexNetBackEnd, CORnetSBackEnd, BasicBlock, ResNetBackEnd, ResNet18
from vone.params import generate_gabor_param
import numpy as np

#Changed to match tinyimagenet
def VOneNet(sf_corr=0.75, sf_max=11.3, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode=None, noise_scale=0.286, noise_level=0.071, k_exc=23.5,
            model_arch='resnet50', image_size=64, visual_degrees=2, ksize=31, stride=2, num_classes=200,
            div_norm=True, vonenet_on=True, color_channels = 1, use_full_image_net=0, restore_path=None, map_location=None):

    #Troubleshooting passing arguments, not relevant.
    map_loc = map_location
    res_path = str(restore_path)
    vone = bool(vonenet_on)
    div = bool(div_norm)
    col = int(color_channels)
    ufin = bool(use_full_image_net)

    #Changes image size depending on type of Image Net
    if ufin == 1:
        image_size = 224
        num_classes = 1000

    #Prints important model properties
    print(f'VOne Net: {vonenet_on}')
    print(f'Divisive Normalization: {div_norm}')
    print(f'Color Channels: {color_channels}')
    print(f'Visual Degrees: {visual_degrees}')
    print(f'Restore Path is: {res_path}')
    print(f'Map location is: {map_loc}')
    print(f'Full imagenet: {ufin}')
    print(f'Image size:{image_size}')
    # Check if using Vonenet, if not you are just loading a basic cnn, likely resnet
    if vone:
        print('Constructing VOneNet')
        # calculate total channels
        out_channels = simple_channels + complex_channels
        
        # generate parameters for channels (receptive fields) from gabor function in params.py
        sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

        gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
        arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


        # Conversions related to pixel size of images
        ppd = image_size / visual_degrees

        sf = sf / ppd
        sigx = nx / sf
        sigy = ny / sf
        #sigx = torch.ones(out_channels)*8
        #sigy = torch.ones(out_channels)*8
        theta = theta/180 * np.pi
        phase = phase / 180 * np.pi
        size = int(image_size//stride)
        

        #Building Voneblock and DN block below are done in modules.py
        # build voneblock frontend using current params
        vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size, color_channels=col)

        # Build divisive norm block                   
        if div:
            dn_block = DivisiveNormBlock(channel_num=out_channels, size=size, ksizeDN=12, use_full_image_net=ufin,
                                         restore_path=res_path, map_location=map_loc)

        if model_arch:
            bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1,
                               bias=False)
            nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out',
                                nonlinearity='relu')

            if model_arch.lower() == 'resnet50':
                print('Model: ', 'VOneResnet50')
                model_back_end = ResNetBackEnd(block=Bottleneck,
                                    layers=[3, 4, 6, 3],
                                    num_classes=num_classes)
            elif model_arch.lower() == 'resnet18':
                print('Model: ', 'VOneResnet18')
                model_back_end = ResNetBackEnd(block=BasicBlock,
                                           layers=[2, 2, 2, 2],
                                           num_classes=num_classes)
            elif model_arch.lower() == 'alexnet':
                print('Model: ', 'VOneAlexNet')
                model_back_end = AlexNetBackEnd(num_classes=num_classes)
            elif model_arch.lower() == 'cornets':
                print('Model: ', 'VOneCORnet-S')
                model_back_end = CORnetSBackEnd(num_classes=num_classes)
            if div:
                print('Model: ', 'VOneNet DN')
                model = nn.Sequential(OrderedDict([
                    ('vone_block', vone_block),
                    ('dn_block', dn_block),
                    ('bottleneck', bottleneck),
                    ('model', model_back_end),
                ]))
            elif not div:
                print('Model: ', 'VOneNet No DN')
                model = nn.Sequential(OrderedDict([
                    ('vone_block', vone_block),
                    ('bottleneck', bottleneck),
                    ('model', model_back_end),
                ]))
        else:
            if div:
                print('Model: ', 'VOneNetDN')
                model = nn.Sequential(OrderedDict([
                    ('vone_block', vone_block),
                    ('dn_block', dn_block),
                    ]))
            elif not div:
                print('Model: ', 'VOneNet')
                model = nn.Sequential(OrderedDict([
                    ('vone_block', vone_block),
                    ]))
        model.gabor_params = gabor_params
        model.arch_params = arch_params
        model.visual_degrees = visual_degrees
        model.image_size = image_size
        if model_arch is None:
            model.identifier = 'vonenet no model_arch'
        else:
            if div:
                model.identifier = 'vone' + model_arch + '_dn'
            else:
                model.identifier = 'vone' + model_arch

    elif not vone:
        model = ResNet18(conv1_stride=1, maxpool1_stride=2, num_classes=num_classes, color_channels = col)
        model.image_size = image_size


    #Return final model
    return model
