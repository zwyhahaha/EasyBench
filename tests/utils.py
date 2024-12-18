from torch.optim import SGD, Adam
from optimizers import OSGM, OSMM
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_optimizer(optimizer_name, params, config):
    learning_rate = config.learning_rate

    if optimizer_name == 'SGD':
        optimizer = SGD(params, lr=learning_rate)
    elif optimizer_name == 'NAG':
        optimizer = SGD(params, lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_name == 'Adam':
        optimizer = Adam(params, lr=learning_rate)
    elif optimizer_name == 'OSGM':
        relax_coef = 1.0 if config.relax_coef is None else config.relax_coef
        optimizer = OSGM(params, lr=learning_rate, relax_coef=relax_coef)
    elif optimizer_name == 'OSMM':
        relax_coef = 1.0 if config.relax_coef is None else config.relax_coef
        beta_lr = 1.0 if config.beta_lr is None else config.beta_lr
        beta = 0.0 if config.beta is None else config.beta
        optimizer = OSMM(params, lr=learning_rate, beta_lr=beta_lr, beta=beta, relax_coef=relax_coef)
    else:
        raise ValueError("Invalid optimizer name")
    return optimizer

def get_scheduler(optimizer, scheduler_name, lr_decay):
    if scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    else:
        scheduler = None
    return scheduler

def get_network_data(config):
    model = config.model
    batch_size = config.batch_size
    if model == 'logreg' or model == 'mlp':
        task = 'MNIST'
        train_loader = DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader
    elif model == 'vgg':
        task = 'CIFAR10'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            pin_memory=True)
        return train_loader, valid_loader
    else:
        raise Exception('Unknown model: {}'.format(model))