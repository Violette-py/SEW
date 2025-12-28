import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

import models
import datasets


def dynamic(x, y):
    """
    Dynamically adjust the noise scale based on clean samples.
    """
    with torch.no_grad():
        net.eval()
        noise = torch.Tensor(np.random.normal(loc=0, scale=data.trigger.scale, size=x.shape)).to(device)
        x_noise = torch.clip(x + noise, 0, 1)
        outputs = net(x_noise)

        outputs = F.normalize(outputs, dim=-1).detach()
        pre_y = outputs[range(y.size(0)), y]
        pre_other_max = outputs[F.one_hot(y, num_classes=data.num_classes) == 0].view(y.size(0), -1).max(-1)[0]
        change = (pre_y - pre_other_max).mean()

        # Update the noise scale
        data.trigger.scale = np.clip(data.trigger.scale + optimizer.param_groups[0]['lr'] * change.item(), 0., 1.)
        net.train()


def train_step(epoch, gap=1):
    """
    Perform a single training step for one epoch.
    """
    net.train()
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets, flags) in enumerate(trainloader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % gap == 0:
            dynamic(inputs[flags == 0], targets[flags == 0])  # Adjust noise scale using clean samples

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            acc = 100. * correct / total
            print(f'TRAIN - Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'CELoss: {train_loss / (batch_idx + 1):.2f} Acc: {acc:.2f}% Noise Scale: {data.trigger.scale:.2f}')


@torch.no_grad()
def test_step(loader):
    """
    Evaluate the model on the given data loader.
    """
    net.eval()
    test_loss, correct, total = 0, 0, 0

    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        outputs = net(inputs)
        test_loss += criterion(outputs, targets).item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc, test_loss


def save_step(epoch, acc):
    """
    Save the model checkpoint for the current epoch.
    """
    global best_acc
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_path, 'final.pth'))

    if sum(acc[:2]) > sum(best_acc):
        print(f'Saving best epoch [{epoch}]...')
        torch.save(net.state_dict(), os.path.join(args.save_path, 'best.pth'))
        best_acc = acc


def train(gap=1):
    """
    Train the model for the specified number of epochs.
    """
    for epoch in range(start_epoch, start_epoch + args.epoch):
        epoch += 1
        train_step(epoch, gap)

        CDA, clean_loss = test_step(testloader)
        print(f'TEST - Epoch: [{epoch}]\t Loss: {clean_loss / len(testloader):.2f}\t CDA: {CDA:.2f}%')

        ASR_train, poison_loss = test_step(trainloader_poison)
        print(f'TEST - Epoch: [{epoch}]\t Loss: {poison_loss / len(trainloader_poison):.2f}\t ASR-train: {ASR_train:.2f}%')

        ASR_test, poison_loss = test_step(testloader_poison)
        print(f'TEST - Epoch: [{epoch}]\t Loss: {poison_loss / len(testloader_poison):.2f}\t ASR-test: {ASR_test:.2f}%')

        cover_CDA, poison_loss = test_step(testloader_cover)
        print(f'TEST - Epoch: [{epoch}]\t Loss: {poison_loss / len(testloader_cover):.2f}\t cover_CDA: {cover_CDA:.2f}%')

        save_step(epoch, [CDA, ASR_train, 0.])
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEW Training')
    
    parser.add_argument('--method', default='sew_post', choices=['sew_pre', 'sew_post'], help='training method of SEW')
    parser.add_argument('--arch', default='vgg16_bn', choices=['vgg16_bn', 'resnet18', 'efficientnet_b3'], help='model architecture')
    parser.add_argument('--dataset', default='Cifar10', choices=['Cifar10', 'Cifar100', 'Tinyimagenet'], help='dataset to train on')
    parser.add_argument('--target_label', default=0, type=int, help='target label for watermarking')
    
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--epoch', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    
    parser.add_argument('--root', default='./data', type=str, help='path to dataset root')
    parser.add_argument('--save_path', default='./results', type=str, help='path to save results')
    
    args = parser.parse_args()
    print(args)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # Dataset setup
    data = datasets.__dict__[args.dataset](root=args.root, batch_size=args.batch_size, num_workers=args.workers, target=args.target_label)

    # Save watermark pattern and mask
    save_image(torch.Tensor(data.trigger.pattern), os.path.join(args.save_path, 'pattern.png'))
    save_image(torch.Tensor(data.trigger.mask), os.path.join(args.save_path, 'mask.png'))

    # Data loader setup
    poison_rate = 0.01 if args.method == 'sew_pre' else 0.02
    cover_rate = 0. if args.method == 'sew_pre' else 0.5
    trainloader, trainloader_poison, testloader, testloader_poison, testloader_cover = data.get_loader(pr=poison_rate, cr=cover_rate)

    # Model setup
    net = models.__dict__[args.arch](num_classes=data.num_classes)
    net.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    best_acc, start_epoch = [], 0

    train(gap=100)
