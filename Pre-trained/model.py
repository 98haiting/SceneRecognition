import torch
from models import *


class models:
    def __init__(self):
        """
        configure resolution for different model
        """
        self.resolution = {
            'vit_h14': 224,
            'vit_l16': 224,
            'vit_b16': 224,
            'regnety_16gf': 224,
            'regnety_32gf': 224,
            'regnety_128gf': 224,
        }

    def Get_model(self, model_name, pretrained=False):
        """

        :param model_name: model to be used
        :return: loaded model and target resolution
        """
        # model = torch.hub.load(repo_or_dir='facebookresearch/swag', model=model_name)
        res = self.resolution[model_name]
        if model_name == 'vit_h14':
            model = vit_h14(pretrained=pretrained)
        elif model_name == 'vit_l16':
            model = vit_l16(pretrained=pretrained)
        elif model_name == 'vit_b16':
            model = vit_b16(pretrained=pretrained)
        elif model_name == 'regnety_16gf':
            model = regnety_16gf(pretrained=pretrained)
        elif model_name == 'regnety_32gf':
            model = regnety_32gf(pretrained=pretrained)
        elif model_name == 'regnety_128gf':
            model = regnety_128gf(pretrained=pretrained)


        if pretrained:
            print(f"loading initial pre-trained weights for {model_name}")

        return model, res