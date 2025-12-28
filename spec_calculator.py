import argparse
import numpy as np
import torch
import torch.nn.functional as F

import models
import datasets

class SEW:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.eps = 0.0  # spec


def dynamic_mean(outputs, y, sew, optimizer):
    """
    Dynamically adjust the spec metric (sew.eps) based on model outputs.
    """
    outputs = F.normalize(outputs, dim=-1).detach()
    pre_y = outputs[range(y.size(0)), y]
    pre_other_max = outputs[F.one_hot(y, num_classes=sew.num_classes) == 0].view(y.size(0), -1).max(-1)[0]
    change = (pre_y - pre_other_max).mean()
    sew.eps = np.clip(sew.eps + optimizer.param_groups[0]['lr'] * change.item(), 0.0, 1.0)


@torch.no_grad()
def calculate_spec(model, wm_loader, device, sew, optimizer, epoch):
    """
    Calculate the spec metric (sew.eps) using the dynamic_mean function.
    """
    model.eval()
    for epoch in range(0, epoch):
        epoch += 1
        for batch_idx, (x, y, _) in enumerate(wm_loader):
            if batch_idx > 10:
                break
            x, y = x.to(device), y.to(device)
            x_eps = torch.clip(x + torch.normal(0, sew.eps + 1e-7, size=x.shape).to(device), 0.0, 1.0)
            x_eps = x_eps.to(device, dtype=torch.float32)
            
            outputs_eps = model(x_eps)
            dynamic_mean(outputs_eps, y, sew, optimizer)
        print(f'EVAL - Epoch: [{epoch}][{batch_idx}/{len(wm_loader)}]\t eps3: {sew.eps:.4f}\t lr: {optimizer.param_groups[0]["lr"]:.8f}\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate spec metric for SEW.')
    
    parser.add_argument('--method', default='sew_post', choices=['sew_pre', 'sew_post'], help='training method of SEW')
    parser.add_argument('--arch', default='vgg16_bn', choices=['vgg16_bn', 'resnet18', 'efficientnet_b3'], help='model architecture')
    parser.add_argument('--dataset', default='Cifar10', choices=['Cifar10', 'Cifar100', 'Tinyimagenet'], help='dataset to train on')
    parser.add_argument('--target_label', default=0, type=int, help='target label for watermarking')
    
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate to update spec')
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--epoch', default=400, type=int, help='number of epochs to calculate spec')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    
    parser.add_argument('--root', default='./data', type=str, help='path to dataset root')
    parser.add_argument('--load_path', default='./results/best.pth', type=str, help='path to load the model checkpoint')
    
    args = parser.parse_args()
    print(args)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Dataset setup
    data = datasets.__dict__[args.dataset](root=args.root, batch_size=args.batch_size, num_workers=args.workers, target=args.target_label)
    poison_rate = 0.01 if args.method == 'sew_pre' else 0.02
    cover_rate = 0. if args.method == 'sew_pre' else 0.5
    _, trainloader_poison, _, _, _ = data.get_loader(pr=poison_rate, cr=cover_rate)

    # Model setup
    model = models.__dict__[args.arch](num_classes=data.num_classes).to(device)
    model.load_state_dict(torch.load(args.load_path, map_location=device))
    model.eval()

    # SEW and optimizer setup
    sew = SEW(num_classes=data.num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Calculate spec
    print("Starting spec calculation...")
    calculate_spec(model, trainloader_poison, device, sew, optimizer, args.epoch)
    print(f"Final spec (eps): {sew.eps:.4f}")