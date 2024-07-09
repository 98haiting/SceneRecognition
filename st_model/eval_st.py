import argparse
import os
import torch, gc
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net
from torch.utils.data import Subset


def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def read_image(type):
    if type == 'style':
        root = args.style
        paths = os.listdir(root)
        subset_indices = None

    if type == 'content':
        if args.data_type == 'train':
            root = './input/content/places365_train_standard.txt'
        elif args.data_type == 'val':
            root = './input/content/places365_val.txt'
        with open(root, 'r') as f:
            filepaths = f.readlines()

        # subset_size = 3650
        subset_indices = torch.randperm(len(filepaths))[:args.subset_size]
        paths = Subset(filepaths, subset_indices)


    return paths, subset_indices

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default='input/content/val_256/',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default='input/style/ODOR3-testset',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default=1)
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='model/decoder_iter_160000.pth')
parser.add_argument('--transform', type=str, default='model/transformer_iter_160000.pth')
parser.add_argument('--data_type', type=str, default='val', help='The type of the training data')
parser.add_argument('--subset_size', type=int, default=3650, help='The size of the subset of the data')

# Additional options
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
transform = net.Transform(in_planes=512)
vgg = net.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

content_paths, subset_indices = read_image('content')
style_paths, _ = read_image('style')
torch.save(subset_indices, args.output + f'/{args.data_type}_indices.pth')
i = 0
for idx in range(len(content_paths)):
    i += 1
    content_path = content_paths[idx].split(' ')[0]
    content = content_tf(Image.open(args.content + content_path).convert('RGB'))
    content = content.to(device).unsqueeze(0)

    style = style_tf(Image.open(args.style + '/' + style_paths[idx % len(style_paths)]).convert('RGB'))
    style = style.to(device).unsqueeze(0)

    with torch.no_grad():
        for x in range(args.steps):
            print('iteration ' + str(x))

            Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
            Content5_1 = enc_5(Content4_1)

            Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
            Style5_1 = enc_5(Style4_1)

            content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))

            content.clamp(0, 255)

        content = content.cpu()

        # training
        if args.data_type == 'train':
            content_name = splitext(content_path)
            content_files = content_name[0].split('/')

            output = args.output
            for content_file in content_files[1:-1]:
                if not os.path.exists(output + '/' + content_file):
                    os.mkdir(output + '/' + content_file)
                    output = output + '/' + content_file
                else:
                    output = output + '/' + content_file
            output_name = '{:s}/{:s}{:s}'.format(output, content_files[-1], content_name[-1])

        # validation
        elif args.data_type == 'val':
            output_name = '{:s}/{:s}'.format(args.output, content_path)
            print("file numer: ", i)
        # content_name = splitext(content_path)
        # content_files = content_name[0].split('/')
        #
        # output = args.output
        # for content_file in content_files[1:-1]:
        #     if not os.path.exists(output + '/' + content_file):
        #         os.mkdir(output + '/' + content_file)
        #         output = output + '/' + content_file
        #     else:
        #         output = output + '/' + content_file
        # output_name = '{:s}/{:s}{:s}'.format(output, content_files[-1], content_name[-1])
        # validation
        # output_name = '{:s}/{:s}'.format(args.output, content_path)

        save_image(content, output_name)
    gc.collect()
    torch.cuda.empty_cache()
