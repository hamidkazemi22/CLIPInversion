import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_loaders(batch_size=256, n_workers=4, dataset_name='cifar10', return_dataset=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100
    train_dataset = dataset(f'data/datasets/{dataset_name}', download=True,
                            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=n_workers)
    test_dataset = dataset(f'data/datasets/{dataset_name}', download=True, train=False,
                           transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=n_workers)
    if return_dataset:
        return train_loader, test_loader, train_dataset, test_dataset
    return train_loader, test_loader


def get_imagenet(batch_size=256, n_workers=4, path='data/datasets/ILSVRC2012/{}', shuffle=True):
    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), ])

    eval_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(), ])

    train_dataset = datasets.ImageFolder(root=path.format('train'),
                                         transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=path.format('val'),
                                        transform=eval_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               num_workers=n_workers, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              num_workers=n_workers, shuffle=True, pin_memory=True)
    return train_loader, test_loader


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


class GuassianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super(GuassianNoise, self).__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, img):
        out = img + torch.randn(img.size()) * self.std + self.mean
        out = torch.clamp(out, 0., 1.)
        return out


def train_step(loader, model_md, loss_fn, opt, epoch_n, scheduler=None, normal_fn=None,
               modify_fn=None, file=None):
    model_md.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (image, label) in enumerate(loader):
        image = image.cuda()
        label = label.cuda()
        opt.zero_grad()
        image = modify_fn(image, label) if modify_fn else image
        image = normal_fn(image) if normal_fn else image
        output = model_md(image)
        preds = torch.argmax(output, -1)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()
        running_loss += loss.item() * image.shape[0]
        running_corrects += torch.sum(preds == label)
        total += image.shape[0]
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        end = '\n' if i == (len(loader) - 1) else '\r'
        print(f'epoch: {epoch_n:04d}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, {i + 1:04d}/{len(loader)}',
              end=end)
    scheduler.step() if scheduler else None


def test_step(test_loader, model, loss_fn, normal_fn=None, modify_fn=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (image, label) in enumerate(test_loader):
        image = image.cuda()
        label = label.cuda()
        image = modify_fn(image, label) if modify_fn else image
        image = normal_fn(image) if normal_fn else image
        with torch.no_grad():
            output = model(image)
            loss = loss_fn(output, label)
        preds = torch.argmax(output, 1)
        running_loss += loss.item() * image.shape[0]
        running_corrects += torch.sum(preds == label.data)
        total += image.shape[0]
        end = '\n' if i == (len(test_loader) - 1) else '\r'

        loss = running_loss / total
        accuracy = running_corrects.double() / total
        print((
            f'Test Loss: {loss:.4f} Test Acc: {accuracy:.4f}, {i + 1:02d}/{len(test_loader)}'),
            end=end)
    accuracy = running_corrects.double() / total
    return accuracy


def adv_test_step(test_loader, model, loss_fn, revertor=None, normal_fn=None,
                  modify_fn=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for i, (image, label) in enumerate(test_loader):
        image = image.cuda()
        label = label.cuda()
        # if i == 0:
        #     plt.figure()
        #     im = image[0].detach().cpu().numpy()
        #     plt.imshow(np.moveaxis(im, 0, -1))
        #     plt.savefig('images/asli.png')
        image = modify_fn(image, label) if modify_fn else image
        if revertor is not None:
            image = image + revertor[label]
        # import matplotlib.pyplot as plt
        # if i == 0:
        #     plt.figure()
        #     im = image[0].detach().cpu().numpy()
        #     plt.imshow(np.moveaxis(im, 0, -1))
        #     plt.savefig('images/adv.png')
        image = normal_fn(image) if normal_fn else image
        with torch.no_grad():
            output = model(image)
            loss = loss_fn(output, label)
        preds = torch.argmax(output, 1)
        running_loss += loss.item() * image.shape[0]
        running_corrects += torch.sum(preds == label.data)
        total += image.shape[0]
        end = '\n' if i == (len(test_loader) - 1) else '\r'

        loss = running_loss / total
        accuracy = running_corrects.double() / total
        print((
            f'Test Loss: {loss:.4f} Test Acc: {accuracy:.4f}, {i + 1:02d}/{len(test_loader)}'),
            end=end)
    accuracy = running_corrects.double() / total
    return accuracy


def freeze_module(module: nn.Module, reverse=False):
    for param in module.parameters():
        param.requires_grad = reverse


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def get_trainable_params(module: nn.Module):
    trainable_params = filter(lambda p: p.requires_grad, module.parameters())
    return trainable_params


def get_optimizer(lr, model=None, params=None):
    trainable_params = params if params else get_trainable_params(model)
    optimizer = torch.optim.SGD(trainable_params,
                                lr=lr, momentum=0.9,
                                dampening=0, weight_decay=1e-4,
                                nesterov=True)
    return optimizer


def params_num(module: nn.Module):
    return len(list(get_trainable_params(module)))


def make_pgd(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn, label, eps, step_size=2 / 255,
             iters=10):
    # copy_model = copy.deepcopy(model)
    copy_model = model
    copy_model.eval()
    copy_image = image.detach().clone()
    # freeze_module(copy_model)
    copy_image.requires_grad = True
    for step in range(iters):
        output = normal_fn(copy_image)
        output = copy_model(output)
        loss = loss_fn(output, label)
        loss.backward()
        adv_image = copy_image + step_size * copy_image.grad.sign()
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image.data = torch.clamp(image.data + perturb.data, 0, 1)

    # del copy_model
    return copy_image


def make_pgd_v2(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn, label, eps, step_size=2 / 255,
                iters=10):
    model.eval()
    # freeze_module(model)
    copy_image = image.detach().clone()
    copy_image.requires_grad = True
    for step in range(iters):
        output = normal_fn(copy_image)
        output = model(output)
        loss = loss_fn(output, label)
        loss.backward()
        adv_image = copy_image + step_size * copy_image.grad.sign()
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image.data = torch.clamp(image.data + perturb.data, 0, 1)
    return copy_image


def make_target_pgd(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn,
                    target_label, eps, iters=10):
    copy_model = copy.deepcopy(model)
    copy_image = image.clone().detach()
    freeze_module(copy_model)
    for step in range(iters):
        copy_image.requires_grad = True
        output = normal_fn(copy_image)
        output = copy_model(output)
        loss = loss_fn(output, target_label)
        loss.backward()
        adv_image = copy_image - eps * copy_image.grad.sign()
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image = image + perturb
        copy_image.detach_()
        copy_image.clamp_(0, 1)

    del copy_model
    return copy_image


def make_adv(model: nn.Module, image: torch.Tensor, normal_fn, loss_fn, label, eps,
             lr=0.1):
    copy_model = copy.deepcopy(model)
    copy_image = image.clone().detach()
    freeze_module(copy_model)
    for step in range(10):
        copy_image.requires_grad = True
        output = normal_fn(copy_image)
        output = copy_model(output)
        loss = loss_fn(output, label)
        loss.backward()
        adv_image = copy_image + lr * copy_image.grad
        perturb = torch.clamp(adv_image - image, -eps, +eps)
        copy_image = image + perturb
        copy_image.detach_()
        copy_image.clamp_(0, 1)

    del copy_model
    return copy_image


def create_file(name):
    try:
        os.makedirs(name)
    except:
        pass


def get_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    return num_params


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


class LogPrint:
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'w')

    def log(self, s):
        sys.stdout.write(s)
        self.file.write(s)

    def close(self):
        self.file.close()

    def flush(self):
        self.file.flush()
        sys.stdout.flush()


def make_image(image: torch.Tensor):
    batch_size, c, h, w = image.shape
    flattened = image.view(batch_size, -1)
    batch_min, batch_max = torch.min(flattened, 1, keepdim=True)[0], torch.max(flattened, 1, keepdim=True)[0]
    flattened -= batch_min
    flattened /= torch.clamp(batch_max - batch_min, min=1e-5)
    return flattened.view(batch_size, c, h, w)


def gray_scale(image):
    return torch.mean(image, dim=1, keepdim=True)


def make_resnet_sequential(model: nn.Module, normalization=None):
    modules = []
    if normalization is not None:
        modules.append(normalization)

    for name, module in model.named_children():
        modules.append(module)
    modules.insert(-1, nn.Flatten())
    out = nn.Sequential(*modules)
    return out


def make_resnet_complete_sequential(base, normalization=None):
    modules = []
    if normalization is not None:
        modules.append(modules)
    for name, module in base.named_children():
        if isinstance(module, nn.Sequential):
            modules.extend(module)
        else:
            modules.append(module)
    modules.insert(-1, nn.Flatten())
    model = nn.Sequential(*modules)
    return model


def zero_grad(image):
    if image.grad is not None:
        if image.grad.grad_fn is not None:
            image.grad.detach_()
        else:
            image.grad.requires_grad_(False)
        image.grad.data.zero_()


class Logger:
    def __init__(self, out_dir, log_name, resume=False):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.fpath = os.path.join(out_dir, log_name)
        if not resume:
            with open(self.fpath, "w") as f:
                f.truncate()

    def log(self, content, end='\n'):
        with open(self.fpath, "a") as f:
            f.write(f'{str(content)}{end}')


def get_ds_info(dataset):
    if dataset == 'cifar10':
        return {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2023, 0.1994, 0.2010],
            'classes': ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        }
    raise Exception('the dataset info was not found.')


def print_message(message, it, last_it):
    end = '\r'
    if it == last_it:
        end = '\n'
    print(message, end=end)


class Scale(nn.Module):
    def __init__(self, size, mode='bicubic'):
        super(Scale, self).__init__()
        self.mode = mode
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=(self.size, self.size), mode=self.mode)


