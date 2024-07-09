import torch
from torch.utils.data import Dataset
import torchvision as tv
import torch.utils.data as data_utils
from PIL import Image
from skimage.color import gray2rgb
from skimage.io import imread

train_mean = [0.485, 0.456, 0.406]
train_std = [0.229, 0.224, 0.225]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode, resolution, args=None):

        self.resolution = resolution
        self.args = args

        if mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.Resize(self.resolution, interpolation=tv.transforms.InterpolationMode.BICUBIC, ),
                tv.transforms.CenterCrop(self.resolution),
                tv.transforms.ToTensor(),  # PIL Image -> tensor
                tv.transforms.Normalize(train_mean, train_std),  # normalize tensor image
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomRotation(degrees=90)
            ])
            if args.stylized:
                subset_indices = torch.load('./data/trainset/output/train_indices.pth')
                self.samples = data_utils.Subset(data, subset_indices)
                self.root = './data/trainset/output/'
            elif args.train_art:
                self.samples = data
                self.root_museum = '/home/janus/iwi5-datasets/fragrant-places/'
                self.root_wiki = '/home/janus/iwi5-datasets/wikidata/'
            else:
                self.samples = data
                #self.root = './data/data_256/'
                self.root = './data/data_large/'
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.Resize(self.resolution, interpolation=tv.transforms.InterpolationMode.BICUBIC,),   # tensor -> PIL Image
                tv.transforms.CenterCrop(self.resolution),
                tv.transforms.ToTensor(),         # PIL Image -> tensor
                tv.transforms.Normalize(train_mean, train_std)   # normalize tensor image
            ])

            if args.stylized:
                subset_indices = torch.load('./data/valset/output/val_indices.pth')
                self.samples = data_utils.Subset(data, subset_indices)
                self.root = './data/valset/output/'
            elif args.train_art:
                self.samples = data
                self.root_museum = '/home/janus/iwi5-datasets/fragrant-places/'
                self.root_wiki = '/home/janus/iwi5-datasets/wikidata/'
            else:
                self.samples = data
                self.root = './data/val_large/'
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        if not self.args.train_art:
            content = self.samples[index].split(' ')
            image = imread(self.root + content[0])
            if len(image.shape) == 2:
                image = gray2rgb(image)
            image = self._transform(image)
            label = torch.tensor(int(content[1].strip('\n')))
        else:
            content = self.samples.iloc[index]
            image_name = content['Image Filename']
            label = content['Label']

            if "wikidata" in image_name:
                filepath = self.root_wiki + image_name
            else:
                filepath = self.root_museum + image_name

            img = Image.open(filepath).convert('RGB')
            image = self._transform(img)

            if type(label) == float:
                label = content['Label']
            else:
                label = torch.tensor(int(label))

        return image, label

class TestDataset(Dataset):
    def __init__(self, data, resolution):
        # get rid of some images that are not available
        self.samples = data
        self.root_ODOR = '/home/janus/iwi5-datasets/p/'
        self.root_museum = '/home/janus/iwi5-datasets/fragrant-places/'
        self.root_wiki = '/home/janus/iwi5-datasets/wikidata/'
        self.resolution = resolution

        self._transform = tv.transforms.Compose([
            # tv.transforms.ToPILImage(),        # tensor -> PIL Image
            tv.transforms.ToTensor(),
            tv.transforms.Resize(self.resolution, antialias=True,
                                 interpolation=tv.transforms.InterpolationMode.BICUBIC,),   # tensor -> PIL Image
            tv.transforms.CenterCrop(self.resolution),
            tv.transforms.Normalize(train_mean, train_std)   # normalize tensor image
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        content = self.samples.iloc[index]
        Image_name = content['Image Filename']
        Label = content['Label']

        if "wikidata" in Image_name:
            filepath = self.root_wiki + Image_name
        elif "Artwork" in Image_name:
            filepath = self.root_ODOR + Image_name
        else:
            filepath = self.root_museum + Image_name

        img = Image.open(filepath).convert('RGB')
        image = self._transform(img)

        if type(Label) == float:
            Label = content['Label']
        else:
            Label = torch.tensor(int(Label))

        return image, Label




