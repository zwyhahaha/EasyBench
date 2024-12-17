import wandb
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.utils import generate_least_squares_data
from tests.utils import get_optimizer, get_scheduler
from models.function import rosenbrock, rastrigin, least_squares

def test_function(config,seed=42):
    batches = config.batches
    epochs = config.epochs
    input_dim = config.input_dim
    task = config.task # ['function', 'network', 'llm']
    model = config.model # {'function': ['rosenbrock', 'rastrigin', 'least_squares'], 'network': ['mlp'], 'llm': ['llm']}
    optimizer_name = config.optimizer # 'SGD', 'NAG', 'Adam', 'OSGM', 'OSMM'
    scheduler_name = config.scheduler # 'None', 'ExponentialLR'
    lr_decay = config.lr_decay

    assert task == 'function'
    assert model in ['rosenbrock', 'rastrigin', 'least_squares']
    assert batches == 1

    wandb.run.name = f"{model}_{optimizer_name}_{config.learning_rate}"

    if model == 'least_squares':
        assert input_dim <= 1000
        X, y = generate_least_squares_data(num_samples=1000, num_features=input_dim, \
                                    noise=0., seed=seed, lib=torch)
    
    w = torch.randn(input_dim, 1, requires_grad=True)
    optimizer = get_optimizer(optimizer_name, [w], config)
    scheduler = get_scheduler(optimizer, scheduler_name, lr_decay)

    for epoch in range(epochs):
        loss_epoch = 0
        for _ in range(batches):
            def closure():
                if model == 'rosenbrock':
                    loss = rosenbrock(w)
                elif model == 'rastrigin':
                    loss = rastrigin(w)
                elif model == 'least_squares':
                    loss = least_squares(w, X, y)
                return loss
            loss = closure()
            optimizer.zero_grad()
            loss.backward()
            if optimizer_name in ['OSGM', 'OSMM']:
                optimizer.step(closure)
            else:
                optimizer.step()
            loss_epoch += loss.item()
        if scheduler is not None:
            scheduler.step()
        wandb.log({'loss': loss_epoch / batches})