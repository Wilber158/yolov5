from torch.utils.data import Dataset
from PIL import Image
import os, argparse, time, subprocess, io, shlex, pickle, pprint
import pandas as pd
import numpy as np
import scipy.ndimage as nd
from tqdm import tqdm
import math
import fire
import glob
import torch.distributed
import xlsxwriter
import matplotlib.pyplot as plt
import matplotx
import imageio

parser = argparse.ArgumentParser(description='Generate Stimulus')
parser.add_argument('--generate_BO_optim_test_set', default=False, type=bool)
parser.add_argument('--generate_BO_standard_test_set', default=False, type=bool)
parser.add_argument('--orientation_divisions', default=12, type=int)
parser.add_argument('--output_path', default=None)
parser.add_argument('--visual_degrees', default=8, type=int)
parser.add_argument('--stim_size', default=224, type=int)
parser.add_argument('--sqr_deg', default=4, type=int)
FLAGS, FIRE_FLAGS = parser.parse_known_args()

def grating_kernel(frequency,  sigma_x, sigma_y, theta=0, offset=0, ks=61):

    w = ks // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)
    x, y = torch.meshgrid(grid_val, grid_val)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = torch.zeros(y.shape)
    g[:] = torch.cos(2 * np.pi * frequency * rotx + offset)

    return g


def center_surround(grating_center=0, grating_surround=0, center_radius=.2, surround_radius=0.6, stim_size=65,
                    stim_type=0):
    x_c = np.linspace(-1, 1, stim_size, endpoint=True)
    xx_c, yy_c = np.meshgrid(x_c, x_c)
    r_c = np.sqrt(xx_c ** 2 + yy_c ** 2)
    if stim_type == 2:
        circle = grating_center * (r_c < center_radius) + grating_surround * (
                    (r_c > center_radius) & (r_c < surround_radius))
    elif stim_type == 0:
        circle = grating_center * (r_c < center_radius)
    elif stim_type == 1:
        circle = grating_surround * ((r_c > center_radius) & (r_c < surround_radius))
    return circle


def generate_bar_stim(length, width, stim_size, figure_color, ground_color=np.array([0, 0, 0]), xshift=0,
                      yshift=0, divisions=10, outline=False, outline_thickness=0.03, outline_color=np.array([0, 0, 0]),
                      posx=0.5, posy=0.5):
    angles = np.linspace(90, -90, divisions, endpoint=False)

    xshift = int((stim_size * xshift) // 2)
    yshift = int((stim_size * yshift) // 2)
    radius_L = int((stim_size * length) // 2)
    radius_W = int((stim_size * width) // 2)

    square_stim = np.zeros((divisions, stim_size, stim_size, 3))
    surround_stim = np.zeros((divisions, stim_size * 2, stim_size * 2, 3))
    surround_stim[:, :, :] = ground_color

    origin = stim_size // 2
    x_lim = np.linspace(2 * origin - radius_W, 2 * origin + radius_W, radius_W * 2, endpoint=False) + xshift
    y_lim = np.linspace(2 * origin - radius_L, 2 * origin + radius_L, radius_L * 2, endpoint=False) + yshift

    idx = 0
    for ang in (angles):
        for x in x_lim:
            for y in y_lim:
                if outline == True:
                    if np.floor(x) < 2 * origin - radius_W + stim_size * outline_thickness + xshift:
                        col = outline_color
                    elif np.floor(x) > 2 * origin + radius_W - stim_size * outline_thickness + xshift:
                        col = outline_color
                    elif np.floor(y) < 2 * origin - radius_L + stim_size * outline_thickness + yshift:
                        col = outline_color
                    elif np.floor(y) > 2 * origin + radius_L - stim_size * outline_thickness + yshift:
                        col = outline_color
                    else:
                        col = figure_color
                else:
                    col = figure_color
                try:
                    surround_stim[idx, int(np.floor(y)), int(np.floor(x))] = col
                except:
                    pass
        surround_rotated = nd.rotate(surround_stim[idx], ang, reshape=False, prefilter=False)
        square_stim[idx] = surround_rotated[int((stim_size - 0.5 * stim_size) - posy * stim_size):int(
            (stim_size + 0.5 * stim_size) - posy * stim_size),
                           int((stim_size - 0.5 * stim_size) - posx * stim_size):int(
                               (stim_size + 0.5 * stim_size) - posx * stim_size)]
        idx += 1
    return square_stim.astype(int)

def combine_bar_stim(stim1, stim2, figure_color_1, figure_color_2, ground_color_1=np.zeros((3)),
                     ground_color_2=np.zeros(3), ground_color_combo=np.zeros((3))):
    divisions = stim1.shape[0]
    stim1 = np.around(np.copy(stim1))
    stim2 = np.around(np.copy(stim2))

    for i in range(3):
        if figure_color_1[i] < ground_color_1[i]:
            stim1[:, :, :, i][stim1[:, :, :, i] < ground_color_1[i]] = figure_color_1[i]
            stim1[:, :, :, i][stim1[:, :, :, i] >= ground_color_1[i]] = 0
        else:
            stim1[:, :, :, i][stim1[:, :, :, i] > ground_color_1[i]] = figure_color_1[i]
            stim1[:, :, :, i][stim1[:, :, :, i] <= ground_color_1[i]] = 0
        if figure_color_2[i] < ground_color_2[i]:
            stim2[:, :, :, i][stim2[:, :, :, i] < ground_color_2[i]] = figure_color_2[i]
            stim2[:, :, :, i][stim2[:, :, :, i] >= ground_color_2[i]] = 0
        else:
            stim2[:, :, :, i][stim2[:, :, :, i] > ground_color_2[i]] = figure_color_2[i]
            stim2[:, :, :, i][stim2[:, :, :, i] <= ground_color_2[i]] = 0

    for d in range(divisions):
        overlapping_pixels = np.where(
            (stim1[d, :, :, 0] == figure_color_1[0]) &
            (stim1[d, :, :, 1] == figure_color_1[1]) &
            (stim1[d, :, :, 2] == figure_color_1[2])
        )
        stim2[d, :, :, :][overlapping_pixels] = figure_color_1
    combined_stim = stim1 + stim2
    for d in range(divisions):
        ground_pixels = np.where(
            (combined_stim[d, :, :, 0] == 0) &
            (combined_stim[d, :, :, 1] == 0) &
            (combined_stim[d, :, :, 2] == 0)
        )
        combined_stim[d, :, :, :][ground_pixels] = ground_color_combo

    return combined_stim.astype(int)

def printtest():
    print('hello')

# xyY color_dict = {'red': np.array([.6, .35, 14]),
#                   'brown': np.array([.6, .35, 2.7]),
#                   'green': np.array([.31, .58, 37]),
#                   'olive': np.array([.31, .58, 6.7]),
#                   'blue': np.array([.16, .08, 6.8]),
#                   'azure': np.array([.16, .08, 1.8]),
#                   'yellow': np.array([.41, .50, 37]),
#                   'beige': np.array([.46, .45, 6.5]),
#                   'violet': np.array([.3, .15, 20]),
#                   'purple': np.array([.3, .15, 3.4]),
#                   'aqua': np.array([.23, .31, 38]),
#                   'cyan': np.array([.23, .31, 7.3]),
#                   'white': np.array([.3, .32, 38]),
#                   'gray': np.array([.3, .32, 8.8]),
#                   'black': np.array([.3, .32, 1.2]),
#                   'light_gray': np.array([.3, .32, 20])}

#https://www.easyrgb.com/en/math.php
'''
def xyY_to_RGB(color_dict):
    for keys in color_dict.keys:
        range = 0 / 255

        var_X = X / 100
        var_Y = Y / 100
        var_Z = Z / 100

        var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
        var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
        var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570

        if (var_R > 0.0031308) var_R = 1.055 * ( var_R ^ ( 1 / 2.4 ) ) - 0.055
    else var_R = 12.92 * var_R
    if (var_G > 0.0031308) var_G = 1.055 * ( var_G ^ ( 1 / 2.4 ) ) - 0.055
    else var_G = 12.92 * var_G
    if (var_B > 0.0031308) var_B = 1.055 * ( var_B ^ ( 1 / 2.4 ) ) - 0.055
    else var_B = 12.92 * var_B

    sR = var_R * 255
    sG = var_G * 255
    sB = var_B * 255
'''
color_dict = {'red': np.array([196.084, 49.116, 17.748]),
                  'brown': np.array([91.809, 17.780, 3.771]),
                  'green': np.array([54.164, 188.413, 24.969]),
                  'olive': np.array([19.433, 85.303, 5.788]),
                  'blue': np.array([31.574, 41.432, 214.595]),
                  'azure': np.array([12.183, 17.849, 117.376]),
                  'yellow': np.array([165.977, 170.160, 28.288]),
                  'beige': np.array([92.777, 68.161, 13.428]),
                  'violet': np.array([206.882, 36.386, 225.503]),
                  'purple': np.array([91.551, 10.065, 100.450]),
                  'aqua': np.array([62.136, 181.170, 193.165]),
                  'cyan': np.array([24.276, 84.137, 90.169]),
                  'white': np.array([158.715, 166.982, 173.276]),
                  'gray': np.array([79.879, 84.373, 87.795]),
                  'black': np.array([26.915, 28.874, 30.365])}

def generate_BO_optimization_set(stim_size = int(FLAGS.stim_size), visual_degrees = FLAGS.visual_degrees,
                                 divisions = FLAGS.orientation_divisions, posy = 0.5, posx = 0.5):
    width = [0.1, .2] #degrees
    length = [.75, 1.5]
    #pos array 0.5 0.5
    ground_color = np.array([118.180, 124.506, 129.323])
    BO_optim_stim_data = pd.DataFrame(columns=['image_id', 'degrees', 'posy', 'posx', 'color', 'orientation', 'width', 'length']) #posx and posy offset
    #image_id degrees posy posx color orientation width length
    DIR = FLAGS.output_path
    color_idx = 0
    print('Constructing...')
    for color_name in tqdm(color_dict.keys()):
        width_idx=0
        for W in width:
            length_idx=0
            for L in length:
                BO_optim_stim_oris = generate_bar_stim(length=L/visual_degrees, width=W/visual_degrees, stim_size=stim_size,
                                                       divisions=divisions, figure_color=color_dict[color_name],
                                                       ground_color=ground_color, outline=False, posx=posx/visual_degrees,
                                                       posy=posy/visual_degrees)
                division_idx = 0
                for d in range(divisions):
                    ID = str(color_idx).zfill(2) + str(width_idx).zfill(2) + str(length_idx).zfill(2) + str(
                        division_idx).zfill(2)
                    BO_optim_stim_img = BO_optim_stim_oris[d].astype(np.uint8)
                    BO_optim_stim_data = BO_optim_stim_data.append(
                        {'image_id': ID, 'degrees':visual_degrees, 'posy':posy, 'posx':posx, 'color': color_name, 'orientation': 180 / 12 * d, 'width': W, 'length': L, },
                        ignore_index=True)
                    file_name = 'BO_optim_stim_' + str(ID) + '.png'
                    imageio.imwrite(DIR + file_name, BO_optim_stim_img)
                    division_idx += 1
                length_idx+=1
            width_idx+=1
        color_idx+=1
    BO_optim_stim_data.to_csv(DIR+'BO_optim_stim_data', index=False)

def generate_BO_standard_test_set(stim_size = int(FLAGS.stim_size), visual_degrees = FLAGS.visual_degrees,
                                 divisions = FLAGS.orientation_divisions, posy = 0.5, posx = 0.5, sqr_deg=FLAGS.sqr_deg):

    ground_color = np.array([118.180, 124.506, 129.323])
    BO_standard_test_stim_data = pd.DataFrame(columns=['image_id', 'degrees', 'posy', 'posx', 'color','orientation', 'polarity', 'side'])
    DIR = FLAGS.output_path
    color_idx = 0
    print('Constructing...')
    for color_name in tqdm(color_dict.keys()):

        for polarity in range(2):
            if polarity == 0:
                ground = ground_color
                figure = color_dict[color_name]
            else:
                ground = color_dict[color_name]
                figure = ground_color

            for side in range(2):
                if side == 0:
                    xshift = 0.5
                else:
                    xshift = -0.5
                BO_standard_test_stim_oris = generate_bar_stim(length=sqr_deg / visual_degrees, width=sqr_deg / visual_degrees,
                                                       stim_size=stim_size,
                                                       divisions=divisions, figure_color=figure,
                                                       ground_color=ground, xshift=xshift, posx=posx/visual_degrees, posy=posy/visual_degrees)
                division_idx = 0
                for d in range(divisions):
                    ID = str(color_idx).zfill(2) + str(polarity).zfill(2) + str(side).zfill(2) + str(
                        division_idx).zfill(2)
                    BO_standard_test_stim_img = BO_standard_test_stim_oris[d].astype(np.uint8)

                    BO_standard_test_stim_data = BO_standard_test_stim_data.append(
                        {'image_id': ID, 'degrees': visual_degrees, 'posy': posy, 'posx': posx, 'color': color_name,
                         'orientation': 180 / 12 * d, 'polarity': polarity, 'side': side, },
                        ignore_index=True)

                    file_name = 'BO_standard_test_stim_' + str(ID) + '.png'
                    imageio.imwrite(DIR + file_name, BO_standard_test_stim_img)


                    division_idx+=1
        color_idx+=1
    BO_standard_test_stim_data.to_csv(DIR+'BO_standard_test_stim_data', index=False)


if FLAGS.generate_BO_standard_test_set == True:
    generate_BO_standard_test_set()
if FLAGS.generate_BO_optim_test_set == True:
    generate_BO_optimization_set()