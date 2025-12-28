"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn
import torch.nn.functional as F

from .ew_layers import EWLinear, EWConv2d


cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=10):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            EWLinear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            EWLinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            EWLinear(4096, num_class)
        )

    def forward(self, x, inspect=False):
        output = self.features(x)
        activation1 = output
        output = F.adaptive_avg_pool2d(output, 1)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        if inspect is False:
            return output
        else:
            return activation1, output

    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

    def disable_ew(self):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.disable()

    def freeze_hidden_layers(self):
        self._freeze_layer(self.features)

    def unfreeze_model(self):
        self._freeze_layer(self.features, freeze=False)
        self._freeze_layer(self.classifier, freeze=False)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn(num_classes=10):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_classes)

def vgg19_bn(num_classes=10):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_class=num_classes)
