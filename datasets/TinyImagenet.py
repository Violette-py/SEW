import os
import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from .trigger import *


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 addnoise=None, target=0, poison_rate=0.01, cover_rate=0.5, trigger=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.addnoise = addnoise
        self.target = target
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.trigger = trigger

        data_dir = os.path.join(root, split)
        self.dataset = ImageFolder(data_dir)
        self.data = [img_path for img_path, _ in self.dataset.samples]
        self.targets = [label for _, label in self.dataset.samples]
        self.num_classes = len(self.dataset.classes)

        np.random.seed(0)
        poison_index = np.random.permutation(len(self.data))[:int(len(self.data) * poison_rate)]
        np.random.seed(None)
        n = int(len(poison_index) * cover_rate)
        self.poison_index, self.cover_index = poison_index[n:], poison_index[:n]

    def __getitem__(self, index):
        flag = 0  # 0: clean, 1: poison, 2: cover
        path = self.data[index]
        target = self.targets[index]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # add trigger pattern
        if index in self.poison_index:
            img = self.trigger(img, cover=False)
            if self.target is None:
                target = (target - 1) % self.num_classes
            else:
                target = self.target
            flag = 1
        elif index in self.cover_index:
            img = self.trigger(img, cover=True)
            flag = 2

        return img, target, flag

    def __len__(self):
        return len(self.data)


class Tinyimagenet(object):
    def __init__(self, root, batch_size, num_workers, target=0, trigger=SEW_trigger(size=64)):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
        self.trigger = trigger
        
        self.num_classes = 200
        self.size = 64

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.size, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    def loader(self, split='train', transform=None, target_transform=None, shuffle=False, poison_rate=0., cover_rate=0., only_poison=False):
        dataset = TinyImageNet(
            root=self.root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            target=self.target,
            poison_rate=poison_rate,
            cover_rate=cover_rate,
            trigger=self.trigger)

        if only_poison:
            dataset.data = np.array(dataset.data)[dataset.poison_index]
            dataset.targets = np.array(dataset.targets)[dataset.poison_index]
            dataset.poison_index = np.arange(len(dataset.data))

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader

    def get_loader(self, pr=0.02, cr=0.5, shuffle=True, raw=False):
        if raw:
            self.transform_train = self.transform_test

        trainloader = self.loader('train', self.transform_train, shuffle=shuffle, poison_rate=pr, cover_rate=cr)
        trainloader_poison = self.loader('train', self.transform_train, shuffle=shuffle, poison_rate=pr, cover_rate=cr, only_poison=True)

        testloader = self.loader('val', self.transform_test, poison_rate=0.)
        testloader_poison = self.loader('val', self.transform_test, poison_rate=1., cover_rate=0.)
        testloader_cover = self.loader('val', self.transform_test, poison_rate=1., cover_rate=1.)

        return trainloader, trainloader_poison, testloader, testloader_poison, testloader_cover
