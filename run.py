"""
using the best hyperparameters to train the model
Given model, task, epochs, batches, input_dim
run all optimizers with the best hyperparameters
"""
import wandb
import yaml
import argparse

parser = argparse.ArgumentParser(description='Train the model with the best hyperparameters.')
parser.add_argument('--model', type=str, required=True, help='The model to use for training')
parser.add_argument('--task', type=str, required=True, help='The task to perform')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Number of batches for training')

args = parser.parse_args()

model = args.model
task = args.task
dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
optimizers = ['SGD', 'NAG', 'Adam', 'OSGM', 'OSMM']

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
for optimizer_name in optimizers:
    wandb.init(project=f'run_{model}_{task}_{dataset}')

    config_path = f'params/{task}/{model}/{optimizer_name}.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    args.optimizer = optimizer_name

    config.update(vars(args))

    config = Config(**config)

    wandb.config.update(config, allow_val_change=True)

    if task == 'function':
        from tests.test_function import test_function
        test_function(wandb.config)
    elif task == 'network':
        from tests.test_network import test_network
        test_network(wandb.config)
    else:
        raise NotImplementedError("Only function task is supported for now")

    wandb.finish()