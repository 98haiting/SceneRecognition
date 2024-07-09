# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:01:58 2020

@author: ZJU
"""

import argparse
from pathlib import Path
import os

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, type=None):
        super(FlatFolderDataset, self).__init__()
        if type == 'content':
            with open(root, 'r') as f:
                samples = f.readlines()
            # self.root = '../data/data_256/'

            # create a subset of the dataset
            subset_size = 5000
            subset_indices = torch.randperm(len(samples))[:subset_size]
            subset_dataset = Subset(samples, subset_indices)
            self.samples = subset_dataset
            self.root = './input/content/data_256'
        elif type == 'style':
            self.root = root
            self.samples = os.listdir(root)
        self.type = type
        self.transform = transform

    def __getitem__(self, index):
        if self.type == 'content':
            content = self.samples[index].split(' ')
            img = Image.open(self.root + content[0]).convert('RGB')

        elif self.type == 'style':
            filepath = self.root + '/' + self.samples[index]
            img = Image.open(filepath).convert('RGB')

        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

contentdir = './input/content/places365_train_standard.txt'
styledir = './input/style/ODOR3-testset'
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default=contentdir,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default=styledir,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--contrastive_weight_c', type=float, default=0.3)
parser.add_argument('--contrastive_weight_s', type=float, default=0.3)
parser.add_argument('--gan_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args('')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda')

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

# gpu memory
print('Memory Usage Before Loading: {:.2f} B'.format(torch.cuda.memory_allocated()))
decoder = net.decoder
vgg = net.vgg

valid = 1
fake = 0
D = net.MultiDiscriminator()
D.to(device)
# this discriminator is 32MB
print('Memory Usage After Loading 1: {:.2f} B'.format(torch.cuda.memory_allocated()))

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, args.start_iter)
network.train()
network.to(device)
# this network is 113-32MB
print('Memory Usage After Loading 2: {:.2f} B'.format(torch.cuda.memory_allocated()))
print("batch size: ", args.batch_size)
content_tf = train_transform()
style_tf = train_transform()


content_dataset = FlatFolderDataset(args.content_dir, content_tf, type='content')
style_dataset = FlatFolderDataset(args.style_dir, style_tf, type='style')

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=int(args.batch_size / 2),
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=int(args.batch_size / 2),
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()},
                              {'params': network.proj_style.parameters()},
                              {'params': network.proj_content.parameters()}], lr=args.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    ######################################################
    content_images_ = content_images[1:]
    content_images_ = torch.cat([content_images_, content_images[0:1]], 0)
    content_images = torch.cat([content_images, content_images_], 0)
    style_images = torch.cat([style_images, style_images], 0)
    ######################################################

    img, loss_c, loss_s, l_identity1, l_identity2, loss_contrastive_c, loss_contrastive_s = network(content_images, style_images, args.batch_size)

    # train discriminator
    loss_gan_d = D.compute_loss(style_images, valid) + D.compute_loss(img.detach(), fake)
    optimizer_D.zero_grad()
    loss_gan_d.backward()
    optimizer_D.step()

    # train generator
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_contrastive_c = args.contrastive_weight_c * loss_contrastive_c
    loss_contrastive_s = args.contrastive_weight_s * loss_contrastive_s
    loss_gan_g = args.gan_weight * D.compute_loss(img, valid)
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1 + loss_contrastive_c + loss_contrastive_s + loss_gan_g

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    writer.add_scalar('loss_contrastive_c', loss_contrastive_c.item(), i + 1)  # attention
    writer.add_scalar('loss_contrastive_s', loss_contrastive_s.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_g', loss_gan_g.item(), i + 1)  # attention

    writer.add_scalar('loss_gan_d', loss_gan_d.item(), i + 1)

    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    # output_dir = args.sample_path
    if (i == 0) or ((i + 1) % 500 == 0):
        output = torch.cat([style_images, content_images, img], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        # output_name = output_dir + '/output' + str(i + 1) + '.jpg'
        # save_image(output, str(output_name), args.batch_size)
        save_image(output, str(output_name))
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
writer.close()
print("end of training")