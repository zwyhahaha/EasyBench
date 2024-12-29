from torch.optim import SGD, Adam
from optimizers import OSGM, OSMM, OSMM2
from torch.optim.lr_scheduler import ExponentialLR
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_optimizer(optimizer_name, params, config):
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay

    if optimizer_name == 'SGD':
        optimizer = SGD(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'NAG':
        optimizer = SGD(params, lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'OSGM':
        relax_coef = 1.5 if not hasattr(config,'relax_coef') else config.relax_coef
        gr_eps = 1e-8 if not hasattr(config,'gr_eps') else config.gr_eps
        dampening = 0.0 if not hasattr(config,'dampening') else config.dampening
        optimizer = OSGM(params, lr=learning_rate, relax_coef=relax_coef,gr_eps=gr_eps, weight_decay=weight_decay,
                         dampening=dampening)
    elif optimizer_name == 'OSMM':
        relax_coef = 1.5 if not hasattr(config,'relax_coef') else config.relax_coef
        beta_lr = 0.1 if not hasattr(config,'beta_lr') else config.beta_lr
        beta = 0.9 if not hasattr(config,'beta') else config.beta
        min_beta = -0.0005 if not hasattr(config,'min_beta') else config.min_beta
        gr_eps = 1e-8 if not hasattr(config,'gr_eps') else config.gr_eps
        stop_step = None if not hasattr(config,'stop_step') else config.stop_step
        dampening = 0.0 if not hasattr(config,'dampening') else config.dampening
        optimizer = OSMM(params, lr=learning_rate, beta_lr=beta_lr, beta=beta, 
                         relax_coef=relax_coef, gr_eps=gr_eps, min_beta=min_beta,
                         stop_step=stop_step, weight_decay=weight_decay,
                         dampening=dampening)
    elif optimizer_name == 'OSMM2':
        relax_coef = 1.5 if not hasattr(config,'relax_coef') else config.relax_coef
        beta_lr = 0.1 if not hasattr(config,'beta_lr') else config.beta_lr
        beta = 0.9 if not hasattr(config,'beta') else config.beta
        min_beta = -0.0005 if not hasattr(config,'min_beta') else config.min_beta
        gr_eps = 1e-8 if not hasattr(config,'gr_eps') else config.gr_eps
        stop_step = None if not hasattr(config,'stop_step') else config.stop_step
        dampening = 0.0 if not hasattr(config,'dampening') else config.dampening
        adagrad = True if not hasattr(config,'adagrad') else config.adagrad
        optimizer = OSMM2(params, lr=learning_rate, beta_lr=beta_lr, beta=beta, 
                         relax_coef=relax_coef, gr_eps=gr_eps, min_beta=min_beta,
                         stop_step=stop_step, weight_decay=weight_decay,
                         dampening=dampening, adagrad=adagrad)
    else:
        raise ValueError("Invalid optimizer name")
    return optimizer

def get_scheduler(optimizer, scheduler_name, lr_decay):
    if scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    else:
        scheduler = None
    return scheduler

def get_network_data(config,seed):
    set_seed(seed)
    model = config.model
    batch_size = config.batch_size
    if model == 'logreg' or model == 'mlp':
        if not hasattr(config, 'dataset') or config.dataset == 'MNIST':
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
        elif 'LIBSVM' in config.dataset:

            task = config.dataset.split('_')[1]
            if 'mnist' in config.dataset:
                data_path = f"data/LIBSVM/{task}.scale.bz2"
            else:
                data_path = f"data/LIBSVM/{task}.scale"
            X, y = load_svmlight_file(data_path)
            X = X.toarray()

            if not np.issubdtype(y.dtype, np.integer):
                unique_labels = np.unique(y)
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y])
            n_classes = len(np.unique(y))
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            y_tensor = convert_to_one_hot(y_tensor, n_classes)
            X_train, X_valid, y_train, y_valid = train_test_split(X_tensor, y_tensor, test_size=0.1,random_state=seed)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataset = TensorDataset(X_valid, y_valid)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, valid_loader
        else:
            raise Exception('Unknown dataset: {}'.format(config.dataset))
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

def convert_to_one_hot(y, num_classes):
    one_hot_encoded = torch.nn.functional.one_hot(y, num_classes=num_classes)
    return one_hot_encoded.float()

def get_data_info(dataset):
    if dataset == 'MNIST':
        input_dim = 28 * 28
        output_dim = 10
    elif 'LIBSVM' in dataset:
        task = dataset.split('_')[1]
        if 'mnist' in dataset:
            data_path = f"data/LIBSVM/{task}.scale.bz2"
        else:
            data_path = f"data/LIBSVM/{task}.scale"
        from sklearn.datasets import load_svmlight_file
        X, y = load_svmlight_file(data_path)
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
    elif dataset == 'CIFAR10':
        input_dim = 3 * 32 * 32
        output_dim = 10
    else:
        raise Exception('Unknown dataset: {}'.format(dataset))
    return input_dim, output_dim