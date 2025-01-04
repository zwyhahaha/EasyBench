import wandb
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils import get_optimizer, get_scheduler, get_network_data, get_data_info, set_seed
from models.network import LogReg, MLP, VGG, vgg16_bn, ResNet18

def test_network(config, seed=42, warmup=False):
    epochs = config.epochs
    task = config.task # ['function', 'network', 'llm']
    model = config.model # {'function': ['rosenbrock', 'rastrigin', 'least_squares'], 'network': ['mlp','vgg','resnet'], 'llm': ['llm']}
    optimizer_name = config.optimizer
    scheduler_name = config.scheduler
    lr_decay = config.lr_decay

    assert task == 'network'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(seed)

    wandb.run.name = f"{optimizer_name}_{config.learning_rate}_seed_{seed}"

    if model == 'logreg':
        input_dim, output_dim = get_data_info(config)
        model = LogReg(input_dim=input_dim, output_dim=output_dim)
    elif model == 'mlp':
        input_dim, output_dim = get_data_info(config)
        model = MLP(input_dim=input_dim, hidden_dim=1000, output_dim=output_dim)
    elif model == 'vgg':
        model = vgg16_bn()
        model.features = torch.nn.DataParallel(model.features)
    elif model == 'resnet':
        model = ResNet18()
    else:
        raise Exception('Unknown model: {}'.format(model))

    model = model.to(device)
    
    train_loader, valid_loader = get_network_data(config, seed)
    optimizer = get_optimizer(optimizer_name, model.parameters(), config)
    scheduler = get_scheduler(optimizer, scheduler_name, lr_decay)

    next_data, next_target = None, None
    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_acc = 0
        if optimizer_name in ['OSMM',"OSMM2"]:
            beta_epoch = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if optimizer_name in ['OSMM',"OSMM2"]:
                beta = optimizer.param_groups[0]['beta'].item()
                beta_epoch += beta
            if optimizer_name in ['OSMM','OSGM',"OSMM2"]:
                # next_data, next_target = next(iter(train_loader))
                # next_data, next_target = next_data.to(device), next_target.to(device)
                # def closure():
                #     next_output = model(next_data)
                #     loss = F.cross_entropy(next_output, next_target)
                #     return loss

                # def closure():
                #     output = model(data)
                #     loss = F.cross_entropy(output, target)
                #     return loss
                # optimizer.step(closure)

                optimizer.step()
            else:
                optimizer.step()
            train_loss += loss.item()
            if config.dataset == 'CIFAR10':
                acc = (output.argmax(dim=1) == target).float().mean()
                train_acc += acc

            if torch.isnan(loss):
                print('Loss is nan')
                if optimizer_name in ['OSMM', 'OSMM2']:
                    wandb.log({'beta': np.nan,
                            'train_loss': np.nan,
                            'valid_loss': np.nan,
                            'train_acc': np.nan,})
                else:
                    wandb.log({'train_loss': np.nan,
                            'valid_loss': np.nan,
                            'train_acc': np.nan,
                            'valid_acc': np.nan})
                wandb.finish()
                return
        
        if scheduler is not None:
            scheduler.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        model.eval()
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += F.cross_entropy(output, target, reduction='sum').item()
                if config.dataset == 'CIFAR10':
                    acc = (output.argmax(dim=1) == target).float().mean()
                    valid_acc += acc
        valid_loss /= len(valid_loader.dataset)
        valid_acc /= len(valid_loader)

        if warmup is False:
            if optimizer_name in ['OSMM',"OSMM2"]:
                wandb.log({'beta': beta_epoch/len(train_loader),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'train_acc': train_acc,
                        'valid_acc': valid_acc})
            else:
                wandb.log({'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'train_acc': train_acc,
                        'valid_acc': valid_acc})
