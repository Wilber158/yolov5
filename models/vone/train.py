from torch.utils.data import Dataset
from PIL import Image
import os, argparse, time, subprocess, io, shlex, pickle, pprint
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import fire
import glob
import torch.distributed
import xlsxwriter
import matplotlib.pyplot as plt
import matplotx
import random
from datetime import datetime
from modules import DivisiveNormBlock

parser = argparse.ArgumentParser(description='ImageNet Training')

## General parameters
parser.add_argument('--vonenet_on', choices=[1, 0], default=1, type=int)
parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains train and val folders')  # IMPORTANT
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')  # IMPORTANT
parser.add_argument('-restore_epoch', '--restore_epoch', default=0, type=int,
                    help='epoch number for restoring model training ')
parser.add_argument('-restore_path', '--restore_path', default=None, type=str,
                    help='path of folder containing specific epoch file for restoring model training')
parser.add_argument('--epoch_name', default=None, type=str)
parser.add_argument('--epoch_restore_number', default=0, type=int)

## Training parameters
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=20, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--optimizer', choices=['stepLR', 'plateauLR'], default='stepLR',
                    help='Optimizer')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=5, type=int,
                    help='after how many epochs learning rate should be decreased by step_factor')
parser.add_argument('--step_factor', default=0.1, type=float,
                    help='factor by which to decrease the learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay ')

## Model parameters
parser.add_argument('--torch_seed', default=0, type=int,
                    help='seed for weights initializations and torch RNG')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet50', 'resnet50_at', 'cornets', 'resnet18'],
                    default='resnet50',
                    help='back-end model architecture to load')
parser.add_argument('--normalization', choices=['vonenet', 'imagenet'], default='vonenet',
                    help='image normalization to apply to models')
parser.add_argument('--visual_degrees', default=2, type=float,  # Try 2 degrees
                    help='Field-of-View of the model in visual degrees')

## VOneBlock parameters
# Gabor filter bank
parser.add_argument('--stride', default=2, type=int,
                    help='stride for the first convolution (Gabor Filter Bank)')
parser.add_argument('--ksize', default=31, type=int,
                    help='kernel size for the first convolution (Gabor Filter Bank)')
parser.add_argument('--simple_channels', default=256, type=int,
                    help='number of simple channels in V1 block')
parser.add_argument('--complex_channels', default=256, type=int,
                    help='number of complex channels in V1 block')
parser.add_argument('--gabor_seed', default=0, type=int,
                    help='seed for gabor initialization')
parser.add_argument('--sf_corr', default=0.75, type=float,
                    help='')
parser.add_argument('--sf_max', default=11.3, type=float,
                    help='')
parser.add_argument('--sf_min', default=0, type=float,
                    help='')
parser.add_argument('--rand_param', choices=[True, False], default=False, type=bool,
                    help='random gabor params')
parser.add_argument('--k_exc', default=23.5, type=float,
                    help='')

# Noise layer
parser.add_argument('--noise_mode', choices=['gaussian', 'neuronal', None],
                    default=None,
                    help='noise distribution')
parser.add_argument('--noise_scale', default=0.286, type=float,
                    help='noise scale factor')
parser.add_argument('--noise_level', default=0.071, type=float,
                    help='noise level')

# Cusom Parsers:

parser.add_argument('--divisive_norm', choices=[1, 0], default=1, type=int)
parser.add_argument('--braintree', choices=[1, 0], default=0, type=int)
parser.add_argument('--use_full_image_net', choices=[1, 0], default=0, type=int)
parser.add_argument('--color_channels', choices=[1, 3], default=1, type=int)
parser.add_argument('--test_arg')


# Robustness Against Corruption Parsers:
parser.add_argument('--c_path', help='path to TinyImageNet-C folder that contains label folders')
parser.add_argument('--array_path', help='path for results of corruption, for use with full_corruption_acc')
parser.add_argument('--verbose', choices=[1, 0], default=0, type=int,
                    help='print progress or not')
parser.add_argument('--windows', choices=[1, 0], default=0, type=int,
                    help='lets me run it on Windows better')
parser.add_argument('--augment', choices=[1, 0], default=0, type=int,
                    help='changes training set to augmented version')

parser.add_argument('--c_name', help='Specify name of array that holds accuracy of the corrupted images')
FLAGS, FIRE_FLAGS = parser.parse_known_args()


def set_gpus(n=2): 
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# if FLAGS.ngpus > 0:
# set_gpus(FLAGS.ngpus)

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
from torchvision import transforms
from __init__ import get_model
import torch.optim as optim

torch.manual_seed(FLAGS.torch_seed)

torch.backends.cudnn.benchmark = True

if FLAGS.ngpus > 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

if FLAGS.normalization == 'vonenet':
    print('VOneNet normalization')
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
elif FLAGS.normalization == 'imagenet':
    print('Imagenet standard normalization')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]


# Loads model using get_model from __init__.py
def load_model():
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    print('Getting VOneNet')
    model = get_model(map_location=map_location, model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.simple_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                      div_norm=bool(FLAGS.divisive_norm), vonenet_on=bool(FLAGS.vonenet_on),
                      color_channels=FLAGS.color_channels,
                      use_full_image_net=bool(FLAGS.use_full_image_net), restore_path=str(FLAGS.restore_path))

    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('We have multiple GPUs detected')
        model = model.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() == 1:
        print('We run on GPU')
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module

    # model = nn.parallel.DistributedDataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    return model


'''
def train(save_train_epochs=.2,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=1,  # how often save model weights
          save_model_secs=720 * 10  # how often save model (in sec)
          ):

    model = load_model()
    #for name, param in model.named_parameters():
       # if param.requires_grad:
           # print(name)

    trainer = ImageNetTrain(model)
    validator = ImageNetVal(model)

    start_epoch = 0
    records = []

    if FLAGS.restore_epoch > 0:
        print('Restoring from previous...')
        ckpt_data = torch.load(os.path.join(FLAGS.restore_path, f'epoch_{FLAGS.restore_epoch:02d}.pth.tar'))
        start_epoch = ckpt_data['epoch']
        print('Loaded epoch: '+str(start_epoch))
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])
        results_old = pickle.load(open(os.path.join(FLAGS.restore_path, 'results.pkl'), 'rb'))
        for result in results_old:
            records.append(result)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }

    # records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)

    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    for epoch in tqdm.trange(start_epoch, FLAGS.epochs + 1, initial=0, desc='epoch'):
        print(epoch)
        data_load_start = np.nan

        data_loader_iter = trainer.data_loader

        for step, data in enumerate(tqdm.tqdm(data_loader_iter, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * nsteps + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    if FLAGS.optimizer == 'plateauLR' and step == 0:
                        trainer.lr.step(results[validator.name]['loss'])
                    trainer.model.train()
                    print('LR: ', trainer.optimizer.param_groups[0]["lr"])

            if FLAGS.output_path is not None:
                if not (os.path.isdir(FLAGS.output_path)):
                    os.mkdir(FLAGS.output_path)

                records.append(results)
                if len(results) > 1:
                    pickle.dump(records, open(os.path.join(FLAGS.output_path, 'results.pkl'), 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                if save_model_secs is not None:
                    if time.time() - recent_time > save_model_secs:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           'latest_checkpoint.pth.tar'))
                        recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path,
                                                           f'epoch_{epoch:02d}.pth.tar'))

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / nsteps
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results = {'meta': {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                           }
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record

            data_load_start = time.time()
'''

best_acc = 0


# https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


initialize_gaussian = 0


# parser.add_argument('--use_full_image_net', choices=[1, 0], default=0, type=int)

# Main training function - called tinytrain because it used to just be for TinyImageNet
def tinytrain():
    epoch_name = str(input('Enter epoch save name:'))
    # Lists to save accuracy, learning rate
    accVec = []
    lrVec = []

    # Dictionary for image ID's, not necessary for full ImageNet
    id_dict = {}

    # Resize image based on Tiny or Full IN
    if FLAGS.use_full_image_net == 1:
        resize = 224
    else:
        resize = 64

    map_location = 'cuda' if FLAGS.ngpus > 0 else 'cpu'

    # TinyIN: Sift through wnids for all image labels, store in id_dict
    if FLAGS.use_full_image_net != 1:
        for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'wnids.txt'))):
            id_dict[line.replace('\n', '')] = i

    # If color_channels is 1 we load in grayscale
    if int(FLAGS.color_channels) == 1:

        # training dataloader + augmentations
        transform_train = transforms.Compose([
            # Random rotation + scaling
            transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Resize(resize),
            # Normalizing
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Grayscale
            torchvision.transforms.Grayscale(num_output_channels=1)
        ])

        # Testing loader + augmentations
        transform_test = transforms.Compose([

            transforms.ToTensor(),
            transforms.Resize(resize),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ])

    else:
        # Same loaders without grayscale...
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load model using get_model function from __init__.py
    model = load_model()

    if FLAGS.use_full_image_net == 1:

        # Not all that important, just loads model parameters and shortens their names
        state = torch.load(os.path.join(FLAGS.restore_path),
                           map_location=map_location)

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state['net'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    for name, param in model.state_dict().items():
        if name != 'module.model.fc.weight' and name != 'module.model.fc.bias':
            param.requires_grad = False
        else:
            print('fc exception')

    # Creates loaders for full IN (same as how Tiago did it in his code)
    if FLAGS.use_full_image_net == 1:
        trainset = ImageNetTrain(model)
        trainloader = trainset.data_loader

        testset = ImageNetVal(model)
        testloader = testset.data_loader

    # Loaders for Tiny IN, custom classses
    else:
        if FLAGS.augment:
            trainset = AugTrainTinyImageNetDataset(id=id_dict, transform=transform_train)
        else:
            trainset = TrainTinyImageNetDataset(id=id_dict, transform=transform_train, braintree=int(FLAGS.braintree))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True)

        testset = TestTinyImageNetDataset(id=id_dict, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False)

    train_loss_vec = []
    train_acc_vec = []
    test_loss_vec = []
    test_acc_vec = []

    # Currently unused function for loading weights - this is done elsewhere currently, but may need to be updated in
    # the future for compatibility/flexibility etc...
    '''
    if (FLAGS.restore_epoch == 1):

        state = torch.load(os.path.join(FLAGS.restore_path),
                              map_location=map_location)

        if FLAGS.divisive_norm == True:
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state['net'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = state['net']

        model.load_state_dict(new_state_dict, strict=False)
    '''

    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    # replace 0.01 as argument for lr with args.lr if using argparse

    # Custom weight decay to stop Divisive Normalization weights from decaying (caused strange activity)
    if FLAGS.divisive_norm and FLAGS.vonenet_on:
        optimizer = optim.SGD([{'params': model.module.dn_block.parameters(), 'weight_decay': 0},
                               {'params': model.module.vone_block.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.bottleneck.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.model.parameters(), 'weight_decay': FLAGS.weight_decay}],
                              lr=FLAGS.lr,
                              momentum=FLAGS.momentum)
    elif FLAGS.vonenet_on and ~FLAGS.divisive_norm:
        optimizer = optim.SGD([{'params': model.module.vone_block.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.bottleneck.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.model.parameters(), 'weight_decay': FLAGS.weight_decay}],
                              lr=FLAGS.lr,
                              momentum=FLAGS.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr, momentum=FLAGS.momentum,
                                    weight_decay=FLAGS.weight_decay)

    # Decrease lr by factor after #patience epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=FLAGS.step_factor)

    # Loop through all epochs
    for epoch in range(FLAGS.epoch_restore_number, FLAGS.epoch_restore_number + FLAGS.epochs):
        print('\nEpoch: %d' % epoch)
        # model in training mode
        model.train()

        # variables to store corresponding values
        train_loss = 0
        correct = 0
        total = 0

        # Pass batches through model
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader))
        print(f'Training Acc: {100. * correct / total} ({correct}/{total})')
        train_loss_vec.append(train_loss)
        train_acc_vec.append(100. * correct / total)

        # Call testing (validation function)
        test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec = \
            test(epoch, test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec,
                 train_acc_vec, epoch_name)


def test(epoch, test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec,
         train_acc_vec, epoch_name):
    global best_acc
    # Put model in evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        scheduler.step(test_loss)

        print(
            f'Validation Acc: {100. * correct / total} ({correct}/{total})')

    test_loss_vec.append(test_loss)
    test_acc_vec.append(100. * correct / total)
    z = 0

    # Troubleshooting parameters values - not relevant.
    checkparams = ['module.dn_block.a', 'module.dn_block.sig', 'module.dn_block.p', 'module.dn_block.thetaD',
                   'module.dn_block.gaussian_bank']
    for name, param in model.named_parameters():
        if name in checkparams:
            print(name)
            print(torch.sum(param.data))

    # Save checkpoint.
    acc = 100. * correct / total
    accVec += [acc]
    lrVec += [get_lr(optimizer)]
    if (acc > best_acc) or (epoch == FLAGS.epoch_restore_number + FLAGS.epochs - 1):
        print('Saving with accuracy', f'{acc}')
        state = {
            'net': model.state_dict(),
            'val_acc': accVec,
            'train_acc': train_acc_vec,
            'epoch': epoch,
            'lr': lrVec
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(FLAGS.output_path, f'Epoch_{epoch}, {epoch_name}.pth'))
        best_acc = acc

        # Create gauss_bank and accuracy plots when done training
        if (epoch == FLAGS.epoch_restore_number + FLAGS.epochs - 2):
            xDN = np.arange(1, len(accVec) + 1, 1)
            labels = [f'{np.max(accVec)}%']
            accVecDN = np.array(accVec)
            yDN = np.full((len(accVec)), np.max(accVec))
            with plt.style.context(matplotx.styles.dufte):
                plt.figure(figsize=(15, 10))
                plt.plot(xDN, accVecDN, label=labels[0])
                plt.plot(xDN, yDN, linestyle='dashed')
                if FLAGS.divisive_norm == True:
                    plt.title(f'Accuracy of VOneNet with Divisive Normalization')
                else:
                    plt.title(f'Accuracy of VOneNet without Divisive Normalization')
                plt.xlabel('Epoch')
                plt.ylabel('% Accuracy')
                matplotx.line_labels()
                plt.savefig(os.path.join(FLAGS.output_path,
                                         f'accuracy_plot, {epoch_name}.png'))

        if (epoch == FLAGS.epoch_restore_number + FLAGS.epochs - 2) and FLAGS.divisive_norm == True:

            gauss_bank = state['net'][f'module.dn_block.gaussian_bank']
            thetaOrder = np.zeros((2, model.module.vone_block.orientations.shape[0]))
            thetaOrder[0, :] = model.module.vone_block.orientations.cpu()
            thetaOrder[1, :] = model.module.vone_block.simple_conv_q0.weight[:, 0, 0, 0].cpu()
            thetaOrder = thetaOrder[0].argsort()

            channels = 15
            step = int((FLAGS.simple_channels + FLAGS.complex_channels) // channels)
            fig, ax = plt.subplots(channels + 1, channels + 1, figsize=(8, 8))

            for x in range(channels + 1):
                for y in range(channels + 1):
                    if (y == 0):
                        v1_k = model.module.vone_block.simple_conv_q0.weight[thetaOrder[x * step], :, :,
                               :].cpu().numpy().mean(axis=0)
                        v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
                        ax[x, y].imshow(v1_k, cmap='gray')
                    elif (x == 0):
                        v1_k = model.module.vone_block.simple_conv_q0.weight[thetaOrder[y * step], :, :,
                               :].cpu().numpy().mean(axis=0)
                        v1_k = v1_k / np.amax(np.abs(v1_k)) / 2 + 0.5
                        ax[x, y].imshow(v1_k, cmap='gray')
                    else:
                        ax[x, y].imshow(gauss_bank[thetaOrder[x * step], thetaOrder[y * step]].cpu())
                    ax[x, y].axes.xaxis.set_visible(False)
                    ax[x, y].axes.yaxis.set_visible(False)
            fig.delaxes(ax[0, 0])
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(os.path.join(FLAGS.output_path, f'Gauss_plot, {epoch_name}.png'))

    return test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec


class ImageNetTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        self.optimizer = torch.optim.SGD(self.model.parameters(), FLAGS.lr, momentum=FLAGS.momentum,
                                         weight_decay=FLAGS.weight_decay)
        if FLAGS.optimizer == 'stepLR':
            self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=FLAGS.step_factor,
                                                      step_size=FLAGS.step_size)
        elif FLAGS.optimizer == 'plateauLR':
            self.lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=FLAGS.step_factor,
                                                                 patience=FLAGS.step_size - 1, threshold=0.01)
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):

        if int(FLAGS.color_channels) == 1:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(FLAGS.in_path, 'train'),
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop(224),
                    transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
                    transforms.RandomHorizontalFlip(0.5),

                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                ]))
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=FLAGS.batch_size,
                                                      shuffle=True,
                                                      num_workers=FLAGS.workers,
                                                      pin_memory=True)
        else:
                dataset = torchvision.datasets.ImageFolder(
                    os.path.join(FLAGS.in_path, 'train'),
                    torchvision.transforms.Compose([
                        torchvision.transforms.RandomResizedCrop(224),
                        transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
                        transforms.RandomHorizontalFlip(0.5),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
                        # torchvision.transforms.Grayscale(num_output_channels=1),
                    ]))
                data_loader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=FLAGS.batch_size,
                                                          shuffle=True,
                                                          num_workers=FLAGS.workers,
                                                          pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()
        if FLAGS.optimizer == 'stepLR':
            self.lr.step(epoch=frac_epoch)
        target = target.to(device)

        output = self.model(inp)

        record = {}
        loss = self.loss(output, target)
        record['loss'] = loss.item()
        record['top1'], record['top5'] = accuracy(output, target, topk=(1, 5))
        record['top1'] /= len(output)
        record['top5'] /= len(output)
        # record['learning_rate'] = self.lr.get_lr()[0]
        record['learning_rate'] = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        record['dur'] = time.time() - start
        return record


class ImageNetVal(object):

    def __init__(self, model):
        self.name = 'val'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self):
        if int(FLAGS.color_channels) == 1:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(FLAGS.in_path, 'val'),
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224),
                    # torchvision.transforms.CenterCrop(64),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                ]))

            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=FLAGS.batch_size,
                                                      shuffle=False,
                                                      num_workers=FLAGS.workers,
                                                      pin_memory=True)
        else:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(FLAGS.in_path, 'val'),
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(224),
                    # torchvision.transforms.CenterCrop(64),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
                    #torchvision.transforms.Grayscale(num_output_channels=1),
                ]))

            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=FLAGS.batch_size,
                                                      shuffle=False,
                                                      num_workers=FLAGS.workers,
                                                      pin_memory=True)
            return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                target = target.to(device)
                output = self.model(inp)

                record['loss'] += self.loss(output, target).item()
                print(self.loss(output, target).item())
                p1, p5 = accuracy(output, target, topk=(1, 5))
                record['top1'] += p1
                record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        # self.filenames = glob.glob("tiny-imagenet-200/val/images/*.JPEG")
        self.filenames = os.listdir(os.path.join(FLAGS.in_path, "val/images"))
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'val/val_annotations.txt'))):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(FLAGS.in_path, "val/images/", self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label
    






class CatImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the cat images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('test')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        #convert to greyscale
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image 
    
class TeapotImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the Teapot images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('test')]
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

class ProcessedCatDataset(Dataset):
    def __init__(self, root_dir, transform=None, level = 'k1'):
        """
        Args:
            root_dir (string): Directory with all the cat images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('cats')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

class ProcessedShadowCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the cat images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('resized') or f.startswith("test")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

class ProcessedTeapotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the teapot images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('teapots')]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image

# Dataset class for training with only cats with labels being the image name
class TrainOnlyWithCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the cat images.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Get a list of filenames if they start with 'n'
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('n')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Adjust the path as needed
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        # Extract label from the filename
        label = self.filenames[idx].split('_')[0]
        if self.transform:
            image = self.transform(image)
        return image, label

class TrainOnlyWithTeapotDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the cat images.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Get a list of filenames if they start with 'n'
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('n')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Adjust the path as needed
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        # Extract label from the filename
        label = self.filenames[idx].split('_')[0]
        if self.transform:
            image = self.transform(image)
        return image, label

class TrainOnlyWithCatTeapotDataset(Dataset):
    def __init__(self, root_dir, id_dict, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images for cats and teapots.
            id_dict (dict): Mapping from class wnids to numeric labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.id_dict = id_dict
        self.filenames = [f for f in os.listdir(root_dir) if f.startswith('n')]
        self.class_dirs = {f.split('_')[0] for f in self.filenames}  # Extract unique class dirs

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        class_dir = self.filenames[idx].split('_')[0]
        
        # Use id_dict to map class_dir to a numeric label
        label = self.id_dict[class_dir]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label









class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None, braintree=1):
        # self.filenames = glob.glob("tiny-imagenet-200/train/*/images/*")
        self.braintree = braintree
        self.filenames = []
        self.filenamesTotal = []
        self.filedirs = []
        for direct in os.listdir(os.path.join(FLAGS.in_path, 'train')):
            self.filenamesTotal = self.filenamesTotal + (
                os.listdir(os.path.join(FLAGS.in_path, 'train', direct, 'images')))
        totalLen = len(self.filenamesTotal)
        dummy = list(self.filenamesTotal)
        self.filenames = self.filenamesTotal
        for i in range(len(self.filenames)):
            self.filedirs = self.filedirs + [
                os.path.join(FLAGS.in_path, 'train', self.filenames[i].split('_')[0], 'images')]
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filedirs[idx] + '/' + self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        if self.braintree == 1:
            label = self.id_dict[self.filedirs[idx].split('/')[7]]
        else:
            label = self.id_dict[self.filedirs[idx].split('/')[6]]
        if self.transform:
            image = self.transform(image)
        return image, label



def tinytest():
    # Determine if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model architecture
    print('Getting VOneNet')
    model = get_model(model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.complex_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                      div_norm=FLAGS.divisive_norm, color_channels=FLAGS.color_channels)

    # Load the trained model weights
    state = torch.load(os.path.join(FLAGS.restore_path, FLAGS.epoch_name), map_location=device)
    model.load_state_dict(state['net'], strict=True)

    # Move the model to the designated device
    model.to(device)

    # Define the transformations
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Grayscale(num_output_channels=1) if FLAGS.color_channels == 1 else None,
    ])

    id_dict = {}
    for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'wnids.txt'))):
        id_dict[line.replace('\n', '')] = i
    # Load the dataset
    testset = TestTinyImageNetDataset(id=id_dict, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            # Move images and labels to the same device as the model
            images, labels = images.to(device), labels.to(device)

            # Perform inference
            outputs = model(images)
            _, predicted = outputs.max(1)

            # Calculate accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Accuracy: {100. * correct / total - 4:.2f}%')


def testModelWithCatImages():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    print('Loading model...')
    model = get_model(model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.complex_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                      div_norm=FLAGS.divisive_norm, color_channels=FLAGS.color_channels).to(device)

    state = torch.load(os.path.join(FLAGS.restore_path, FLAGS.epoch_name), map_location=device)
    model.load_state_dict(state['net'], strict=True)
    model.to(device)

    # Load the cat images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Grayscale(num_output_channels=1) if FLAGS.color_channels == 1 else None,
    ])
    cat_images_dataset = CatImagesDataset(root_dir="/home/wcortez/SurroundSuppression/Surround-Suppresion-for-VOneNet/Regular_Cat_Images", transform=transform)
    cat_images_loader = torch.utils.data.DataLoader(cat_images_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    # Prepare class label mapping
    id_dict = {}
    with open(os.path.join(FLAGS.in_path, 'wnids.txt'), 'r') as f:
        for index, line in enumerate(f):
            id_dict[index] = line.strip()
    # Load human-readable class names
    class_names = {}
    with open("/home/wcortez/Documents/tiny-imagenet-200/words.txt", 'r') as f:  # Assuming you have a "words.txt" mapping wnids to class names
        for line in f:
            wnid, label = line.split('\t')
            class_names[wnid] = label.strip()

    # Evaluate the model on cat images
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images in cat_images_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for idx in predicted:
                wnid = id_dict[idx.item()]
                label = class_names.get(wnid, "Unknown")
                print(f"Predicted class: {label}")
                total += 1
                if ("cat" in label):
                    correct+=1
    
    #print the accuracy
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f'Accuracy: {100. * correct / total:.2f}%')



def testModelWithTeapotImages():

    torch.cuda.empty_cache()
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    print('Loading model...')
    model = get_model(model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.complex_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                      div_norm=FLAGS.divisive_norm, color_channels=FLAGS.color_channels).to(device)

    state = torch.load(os.path.join(FLAGS.restore_path, FLAGS.epoch_name), map_location=device)
    model.load_state_dict(state['net'], strict=True)
    model.to(device)

    # Load the cat images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Grayscale(num_output_channels=1) if FLAGS.color_channels == 1 else None,
    ])
    cat_images_dataset = ProcessedTeapotDataset(root_dir="/home/wcortez/SurroundSuppression/Surround-Suppresion-for-VOneNet/Proccessed_Teapot_Images/k5", transform=transform)
    cat_images_loader = torch.utils.data.DataLoader(cat_images_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    # Prepare class label mapping
    id_dict = {}
    with open(os.path.join(FLAGS.in_path, 'wnids.txt'), 'r') as f:
        for index, line in enumerate(f):
            id_dict[index] = line.strip()
    # Load human-readable class names (if available)
    class_names = {}
    with open("/home/wcortez/Documents/tiny-imagenet-200/words.txt", 'r') as f: 
        for line in f:
            wnid, label = line.split('\t')
            class_names[wnid] = label.strip()

    # Evaluate the model on cat images
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images in cat_images_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for idx in predicted:
                wnid = id_dict[idx.item()]
                label = class_names.get(wnid, "Unknown")
                print(f"Predicted class: {label}")
                total += 1
                if ("teapot" in label):
                    correct+=1
    
    #print the accuracy
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f'Accuracy: {100. * correct / (total):.2f}%')



def testModelWithShadowImages():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(torch.load(os.path.join(FLAGS.restore_path, FLAGS.epoch_name)))
    # Load model
    print('Loading model...')
    model = get_model(model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.complex_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                      div_norm=FLAGS.divisive_norm, color_channels=FLAGS.color_channels).to(device)

    state = torch.load(os.path.join(FLAGS.restore_path, FLAGS.epoch_name), map_location=device)
    model.load_state_dict(state['net'], strict=True)
    model.to(device)

    # Load the cat images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Grayscale(num_output_channels=1) if FLAGS.color_channels == 1 else None
    ])
    cat_images_dataset = ProcessedShadowCatDataset(root_dir="/home/wcortez/SurroundSuppression/Surround-Suppresion-for-VOneNet/240322_cats/240321_old_lightness_cats/Shadow_cats_2", transform=transform)
    cat_images_loader = torch.utils.data.DataLoader(cat_images_dataset, batch_size=FLAGS.batch_size, shuffle=False)


    # Prepare class label mapping
    id_dict = {}
    with open(os.path.join(FLAGS.in_path, 'wnids.txt'), 'r') as f:
        for index, line in enumerate(f):
            id_dict[index] = line.strip()
    # Load human-readable class names
    class_names = {}
    with open("/home/wcortez/Documents/tiny-imagenet-200/words.txt", 'r') as f:  # Assuming you have a "words.txt" mapping wnids to class names
        for line in f:
            wnid, label = line.split('\t')
            class_names[wnid] = label.strip()

    # Evaluate the model on cat images
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images in cat_images_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for idx in predicted:
                wnid = id_dict[idx.item()]
                label = class_names.get(wnid, "Unknown")
                print(f"Predicted class: {label}")
                total += 1
                if ("cat" in label):
                    correct+=1
    
    #print the accuracy
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f'Accuracy: {100. * correct / total:.2f}%')
    









# -------------------------- Functions added for testing Robustness Against Corruption-------------------------
# Note: Add ability to augment training data later
class TinyImageNetCDataset(Dataset):
    def __init__(self, id, corruption_type="*", severity="[1-5]", transform=None):
        self.filenames = glob.glob(os.path.join(FLAGS.c_path, f"{corruption_type}/{severity}/n*/test_[0-9]*"))
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.id_dict[img_path.split('/')[-2]]
        if self.transform:
            image = self.transform(image)
        return image, label


class AugTrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob(f"{FLAGS.in_path}/train/n*/images/n*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, c_chance=0.5):
        if np.random.uniform() < c_chance:
            corruption_type = random.choice(["contrast", "fog", "pixelate", "defocus_blur"])
            severity = random.choice(["1", "2", "3", "4", "5"])
            h, t = os.path.split(self.filenames[idx])
            inpath = os.path.split(h)[0]
            img_path = f"{inpath}/{corruption_type}/{severity}/{t}"
            label = self.id_dict[img_path.split('/')[-4]]
        else:
            img_path = self.filenames[idx]
            label = self.id_dict[img_path.split('/')[-3]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    

# class TinyImageNetCDatasetWin(Dataset):
#     def __init__(self, id, corruption_type="*", severity="[1-5]", transform=None):
#         self.filenames = glob.glob(os.path.join(FLAGS.c_path, f"{corruption_type}/{severity}/n*/test_[0-9]*"))
#         self.transform = transform
#         self.id_dict = id
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = Image.open(img_path).convert('RGB')
#         label = self.id_dict[img_path.split('\\')[-2]]
#         if self.transform:
#             image = self.transform(image)
#         return image, label

def full_corruption_acc(get_clean=False):
    """Gets all corruption data"""
    if not FLAGS.c_name:
        array_name = str(input('Enter output array save name:'))
    else: 
        array_name = FLAGS.c_name
    map_location = 'cuda' if FLAGS.ngpus > 0 else 'cpu'
    print('Getting model')
    epath = os.path.join(FLAGS.restore_path, FLAGS.epoch_name)
    state = torch.load(epath, map_location=map_location)
    save_path = os.path.join(FLAGS.array_path, array_name)
    from os import path
    #code for making sure the array path is valid so it doesn't wait to throw the File Not Found error until AFTER it's done
    try:
        assert (path.exists(save_path)), "array output must be an existing directory"
    except Exception as e:
        print(e)
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state['net'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    print('Loading model...')
    model = get_model(model_arch=FLAGS.model_arch, pretrained=False,
                      visual_degrees=FLAGS.visual_degrees, stride=FLAGS.stride, ksize=FLAGS.ksize,
                      sf_corr=FLAGS.sf_corr, sf_max=FLAGS.sf_max, sf_min=FLAGS.sf_min, rand_param=FLAGS.rand_param,
                      gabor_seed=FLAGS.gabor_seed, simple_channels=FLAGS.simple_channels,
                      complex_channels=FLAGS.complex_channels, noise_mode=FLAGS.noise_mode,
                      noise_scale=FLAGS.noise_scale, noise_level=FLAGS.noise_level, k_exc=FLAGS.k_exc,
                      div_norm=FLAGS.divisive_norm, color_channels=FLAGS.color_channels).to(device)
    # model = get_model(map_location=map_location, model_arch=FLAGS.model_arch, pretrained=False,
    #                   simple_channels=FLAGS.simple_channels, complex_channels=FLAGS.simple_channels, color_channels=FLAGS.color_channels,
    #                   noise_mode=FLAGS.noise_mode, div_norm=FLAGS.divisive_norm).module

    model.load_state_dict(new_state_dict, strict=False)
    result_table = []
    corruption_categories = ["shot_noise", "brightness", "pixelate", "glass_blur", "motion_blur", "impulse_noise",
                             "frost", "jpeg_compression", "contrast", "defocus_blur", "elastic_transform", "snow",
                             "fog", "gaussian_noise", "zoom_blur"]
    severities = np.arange(1, 6, 1)
    for severity in severities:
        for corruption_type in corruption_categories:
            correct, total = corruption_acc(model, corruption_type=corruption_type, severity=severity)
            result_table.append([corruption_type, severity, (float(correct) / float(total) * 100)])
    result_array = np.array(result_table)
    # Get average values
    averages = []
    for ctype in corruption_categories:
        averages.append([ctype, avg_pert(result_array, ctype)])
    avg = np.array(averages)
    np.save(save_path, result_array)
    np.save(save_path + "avg", avg)
    df = pd.DataFrame(avg, index=corruption_categories)
    df2 = pd.DataFrame(result_array)
    df.to_csv(save_path+ "avg")
    df2.to_csv(save_path)
    print("Clean accuracy is ", state["val_acc"][-1])


def corruption_acc(model, corruption_type="*", severity="[1-5]"):
    id_dict = {}
    for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'wnids.txt'))):
        id_dict[line.replace('\n', '')] = i
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(num_output_channels=1)

    ])

    testset = TinyImageNetCDataset(id=id_dict, corruption_type=corruption_type, severity=severity, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=True)


    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(testloader):
            image, label = image.cuda(), label.cuda()
            if FLAGS.verbose:
                print('Label')
                print(label)

            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = model.to(device)

            image = image.to(device)

            outputs = model(image)
            _, predicted = outputs.max(1)
            if FLAGS.verbose:
                print('Predicted')
                print(predicted)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if FLAGS.verbose:
                print('Correct')
                print(correct)

        print(f'Got {correct} / {total} with accuracy {float(correct) / float(total) * 100:.2f}')
    return correct, total


def avg_pert(result_array, corruption_type):
    x = np.where(result_array == corruption_type)[0]
    y = result_array[x, 2].astype(float)
    return np.mean(y)


def clean_accuracy(id_dict, model, verbose=False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = TestTinyImageNetDataset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(testloader):
            image, label = image.cuda(), label.cuda()
            if verbose:
                print('Label')
                print(label)
            outputs = model(image)
            _, predicted = outputs.max(1)
            if verbose:
                print('Predicted')
                print(predicted)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if verbose:
                print('Correct')
                print(correct)

        print(f'Got {correct} / {total} with accuracy {float(correct) / float(total) * 100:.2f}')
    return correct, total


class MixedTinyImageNetDataset(Dataset):
    def __init__(self, regular_dataset, corrupted_dataset, corruption_chance=0.75):
        """
        A dataset that mixes regular and corrupted images.
        :param regular_dataset: The regular dataset.
        :param corrupted_dataset: The corrupted dataset.
        :param corruption_chance: The chance to pick an image from the corrupted dataset.
        """
        self.regular_dataset = regular_dataset
        self.corrupted_dataset = corrupted_dataset
        self.corruption_chance = corruption_chance

    def __len__(self):
        # The length is determined by the regular dataset to ensure we go through all regular images.
        return len(self.regular_dataset)

    def __getitem__(self, idx):
        if np.random.rand() < self.corruption_chance:
            # Choose a random index from the corrupted dataset
            corr_idx = np.random.randint(len(self.corrupted_dataset))
            return self.corrupted_dataset[corr_idx]
        else:
            return self.regular_dataset[idx]

def augmentTinyTrain():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epoch_name = input('Enter epoch save name:')
    id_dict = {line.replace('\n', ''): i for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'wnids.txt')))}
    
    transform = transforms.Compose([
    transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((64, 64))
    ])

    regular_dataset = TrainTinyImageNetDataset(id_dict, transform=transform)
    corrupted_dataset = TinyImageNetCDataset(id_dict, transform=transform)

    # Combining datasets logic is handled here
    combined_data = []
    for _ in range(len(regular_dataset)):
        if np.random.rand() < 0.75:  # Corruption chance
            idx = np.random.randint(0, len(corrupted_dataset))
            combined_data.append(corrupted_dataset[idx])
        else:
            idx = np.random.randint(0, len(regular_dataset))
            combined_data.append(regular_dataset[idx])

    train_loader = torch.utils.data.DataLoader(combined_data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)

    # Model preparation 
    model = load_model()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    if FLAGS.divisive_norm and FLAGS.vonenet_on:
        optimizer = optim.SGD([{'params': model.module.dn_block.parameters(), 'weight_decay': 0},
                               {'params': model.module.vone_block.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.bottleneck.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.model.parameters(), 'weight_decay': FLAGS.weight_decay}],
                              lr=FLAGS.lr,
                              momentum=FLAGS.momentum)
    elif FLAGS.vonenet_on and ~FLAGS.divisive_norm:
        optimizer = optim.SGD([{'params': model.module.vone_block.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.bottleneck.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.model.parameters(), 'weight_decay': FLAGS.weight_decay}],
                              lr=FLAGS.lr,
                              momentum=FLAGS.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr, momentum=FLAGS.momentum,
                                    weight_decay=FLAGS.weight_decay)

    # Decrease lr by factor after #patience epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=FLAGS.step_factor)
    train_loss_vec = []
    train_acc_vec = []
    test_loss_vec = []
    test_acc_vec = []
    # Loop through all epochs
    for epoch in range(FLAGS.epoch_restore_number, FLAGS.epoch_restore_number + FLAGS.epochs):
        print('\nEpoch: %d' % epoch)
        # model in training mode
        model.train()

        # variables to store corresponding values
        train_loss = 0
        correct = 0
        total = 0

        # Pass batches through model
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader))
        print(f'Training Acc: {100. * correct / total} ({correct}/{total})')
        train_loss_vec.append(train_loss)
        train_acc_vec.append(100. * correct / total)

        test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec = \
            test(epoch, test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec,
                 train_acc_vec, epoch_name)
        
def tinytrainCats():
    epoch_name = str(input('Enter epoch save name:'))
    # Lists to save accuracy, learning rate
    accVec = []
    lrVec = []

    # Dictionary for image ID's, not necessary for full ImageNet
    id_dict = {}

    # Resize image based on Tiny or Full IN
    if FLAGS.use_full_image_net == 1:
        resize = 224
    else:
        resize = 64

    map_location = 'cuda' if FLAGS.ngpus > 0 else 'cpu'

    # TinyIN: Sift through wnids for all image labels, store in id_dict
    if FLAGS.use_full_image_net != 1:
        for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'wnids.txt'))):
            id_dict[line.replace('\n', '')] = i

    # If color_channels is 1 we load in grayscale
    if int(FLAGS.color_channels) == 1:

        # training dataloader + augmentations
        transform_train = transforms.Compose([
            # Random rotation + scaling
            transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Resize(resize),
            # Normalizing
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Grayscale
            torchvision.transforms.Grayscale(num_output_channels=1)
        ])

        # Testing loader + augmentations
        transform_test = transforms.Compose([

            transforms.ToTensor(),
            transforms.Resize(resize),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Grayscale(num_output_channels=1),
        ])

    else:
        # Same loaders without grayscale...
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(1.0, 1.2)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load model using get_model function from __init__.py
    model = load_model()

    if FLAGS.use_full_image_net == 1:

        # Not all that important, just loads model parameters and shortens their names
        state = torch.load(os.path.join(FLAGS.restore_path),
                           map_location=map_location)

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state['net'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

    for name, param in model.state_dict().items():
        if name != 'module.model.fc.weight' and name != 'module.model.fc.bias':
            param.requires_grad = False
        else:
            print('fc exception')


    # Loaders for Tiny IN, custom classses

    id_dict = {}  # Maps indices to wnids
    class_names = {}  # Maps wnids to class names like "cat" or "teapot"
    for i, line in enumerate(open(os.path.join(FLAGS.in_path, 'wnids.txt'))):
            id_dict[line.replace('\n', '')] = i
    with open(os.path.join(FLAGS.in_path, 'words.txt'), 'r') as f:
        for line in f:
            wnid, label = line.strip().split('\t')
            class_names[wnid] = label
    

    trainset = TrainOnlyWithCatTeapotDataset('/home/wcortez/SurroundSuppression/Surround-Suppresion-for-VOneNet/Cats & Teapots', id_dict=id_dict, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True)

    testset = TestTinyImageNetDataset(id=id_dict, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False)

    train_loss_vec = []
    train_acc_vec = []
    test_loss_vec = []
    test_acc_vec = []


    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    # replace 0.01 as argument for lr with args.lr if using argparse

    # Custom weight decay to stop Divisive Normalization weights from decaying (caused strange activity)
    if FLAGS.divisive_norm and FLAGS.vonenet_on:
        optimizer = optim.SGD([{'params': model.module.dn_block.parameters(), 'weight_decay': 0},
                               {'params': model.module.vone_block.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.bottleneck.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.model.parameters(), 'weight_decay': FLAGS.weight_decay}],
                              lr=FLAGS.lr,
                              momentum=FLAGS.momentum)
    elif FLAGS.vonenet_on and ~FLAGS.divisive_norm:
        optimizer = optim.SGD([{'params': model.module.vone_block.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.bottleneck.parameters(), 'weight_decay': FLAGS.weight_decay},
                               {'params': model.module.model.parameters(), 'weight_decay': FLAGS.weight_decay}],
                              lr=FLAGS.lr,
                              momentum=FLAGS.momentum)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr, momentum=FLAGS.momentum,
                                    weight_decay=FLAGS.weight_decay)

    # Decrease lr by factor after #patience epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=FLAGS.step_factor)

    # Loop through all epochs
    for epoch in range(FLAGS.epoch_restore_number, FLAGS.epoch_restore_number + FLAGS.epochs):
        print('\nEpoch: %d' % epoch)
        # model in training mode
        model.train()

        # variables to store corresponding values
        train_loss = 0
        correct = 0
        total = 0

        # Pass batches through model
        for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader))
        print(f'Training Acc: {100. * correct / total} ({correct}/{total})')
        train_loss_vec.append(train_loss)
        train_acc_vec.append(100. * correct / total)
        #test every 10 epochs
        if epoch % 10 == 0:
            test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec = \
                test(epoch, test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec,
                    train_acc_vec, epoch_name)

    # Call testing (validation function)
    test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec = \
        test(epoch, test_loss_vec, test_acc_vec, model, testloader, criterion, scheduler, optimizer, accVec, lrVec,
                train_acc_vec, epoch_name)

    


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)


