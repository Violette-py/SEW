import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from .trigger import *


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, addnoise=None,
                 target=0, poison_rate=0.01, cover_rate=0.5, trigger=None):
        super(CIFAR10, self).__init__(root,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)
        
        self.addnoise = addnoise
        self.target = target
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.trigger = trigger

        np.random.seed(0)
        poison_index = np.random.permutation(len(self.data))[:int(len(self.data) * poison_rate)]
        np.random.seed(None)
        n = int(len(poison_index) * cover_rate)
        self.poison_index, self.cover_index = poison_index[n:], poison_index[:n]

    def __getitem__(self, index):
        flag = 0  # 0: clean sample, 1: poison sample, 2: cover sample
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if index in self.poison_index:  # set target label
            img = self.trigger(img, cover=False)
            if self.target is None:
                target = (target - 1) % 10
            else:
                target = self.target
            flag = 1
        if index in self.cover_index:  # keep original label
            img = self.trigger(img, cover=True)
            flag = 2
        return img, target, flag


class Cifar10(object):
    def __init__(self, root, batch_size, num_workers, target=0, trigger=SEW_trigger()):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target

        self.num_classes = 10
        self.size = 32

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.size, 2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trigger = trigger

    def loader(self, split='train', transform=None, target_transform=None, shuffle=False, poison_rate=0., cover_rate=0., only_posion=False):
        train = (split == 'train')
        dataset = CIFAR10(
            root=self.root, train=train, download=True,
            transform=transform, target_transform=target_transform, target=self.target, poison_rate=poison_rate, cover_rate=cover_rate, trigger=self.trigger)

        if only_posion:
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
        trainloader_poison = self.loader('train', self.transform_train, shuffle=shuffle, poison_rate=pr, cover_rate=cr, only_posion=True)

        testloader = self.loader('test', self.transform_test, poison_rate=0.)
        testloader_poison = self.loader('test', self.transform_test, poison_rate=1., cover_rate=0.)
        testloader_cover = self.loader('test', self.transform_test, poison_rate=1., cover_rate=1.)
        return trainloader, trainloader_poison, testloader, testloader_poison, testloader_cover
