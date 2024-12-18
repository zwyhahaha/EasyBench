import wandb
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.utils import get_optimizer, get_scheduler, get_network_data, get_data_info
from models.network import LogReg, MLP, VGG, vgg16_bn

def test_network(config, seed=42):
    epochs = config.epochs
    task = config.task # ['function', 'network', 'llm']
    model = config.model # {'function': ['rosenbrock', 'rastrigin', 'least_squares'], 'network': ['mlp'], 'llm': ['llm']}
    optimizer_name = config.optimizer # 'SGD', 'NAG', 'Adam', 'OSGM', 'OSMM'
    scheduler_name = config.scheduler # 'None', 'ExponentialLR'
    lr_decay = config.lr_decay

    assert task == 'network'
    assert model in ['logreg', 'mlp', 'vgg']

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.device("cuda")
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
    else:
        torch.device("cpu")
        torch.manual_seed(seed)

    wandb.run.name = f"{model}_{optimizer_name}_{config.learning_rate}"

    if model == 'logreg':
        if config.dataset is None:
            model = LogReg(input_dim=28 * 28, output_dim=10)
        else:
            input_dim, output_dim = get_data_info(config.dataset)
            model = LogReg(input_dim=input_dim, output_dim=output_dim)
    elif model == 'mlp':
        model = MLP(input_dim=28 * 28, hidden_dim=1000, output_dim=10)
    elif model == 'vgg':
        model = vgg16_bn()
        model.features = torch.nn.DataParallel(model.features)
    else:
        raise Exception('Unknown model: {}'.format(model))

    if use_cuda:
        model = model.cuda()
    
    train_loader, valid_loader = get_network_data(config)
    optimizer = get_optimizer(optimizer_name, model.parameters(), config)
    scheduler = get_scheduler(optimizer, scheduler_name, lr_decay)

    next_data, next_target = None, None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        if optimizer_name in ['OSMM']:
            beta_epoch = 0
        for data, target in train_loader:
            data, target = Variable(data), Variable(target)
            next_data, next_target = next(iter(train_loader))
            next_data, next_target = Variable(next_data), Variable(next_target)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                next_data, next_target = next_data.cuda(), next_target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if optimizer_name in ['OSMM']:
                beta = optimizer.param_groups[0]['beta'].item()
                beta_epoch += beta
            if optimizer_name in ['OSMM','OSGM']: 
                def closure():
                    next_output = model(next_data)
                    loss = F.cross_entropy(next_output, next_target)
                    return loss
                optimizer.step(closure)
                
            else:
                optimizer.step()
            train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        wandb.log({'train_loss': train_loss / len(train_loader.dataset)})
        if optimizer_name in ['OSMM']:
            wandb.log({'beta': beta_epoch / len(train_loader.dataset)})
        
        model.eval()
        valid_loss = 0
        for data, target in valid_loader:
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                valid_loss += F.cross_entropy(output, target, reduction='sum').item()
        valid_loss /= len(valid_loader.dataset)
        wandb.log({'valid_loss': valid_loss})