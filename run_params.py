"""
using the best hyperparameters to train the model
Given model, task, epochs, batches, input_dim
run all optimizers with the best hyperparameters
"""
import wandb
import yaml
import argparse

parser = argparse.ArgumentParser(description='Train the model with the best hyperparameters.')
parser.add_argument('--model', type=str, default="logreg", help='The model to use for training')
parser.add_argument('--task', type=str, default="network", help='The task to perform')
parser.add_argument('--dataset', type=str, default="LIBSVM_vowel", help='dataset')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Number of batches for training')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--scheduler', type=str, default=None) # ExponentialLR
parser.add_argument('--lr_decay', type=float, default=1.0)
args = parser.parse_args()

model = args.model
task = args.task
dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
optimizers = ['SGD', 'NAG', 'Adam', 'OSMM']
# optimizers = ['OSMM']

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
for optimizer_name in optimizers:
    if args.seed is None:
        seeds = range(1)
    else:
        seeds = [args.seed]
    
    for seed in seeds:
        wandb.init(project=f'run_seeds_{model}_{task}_{dataset}_weight_decay_{args.weight_decay}')

        if dataset is not None:
            config_path = f'params/{task}/{model}/{dataset}_{batch_size}/{optimizer_name}.yaml'
        else:
            config_path = f'params/{task}/{model}/{optimizer_name}.yaml'
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if 'gr_eps' in config:
            config['gr_eps'] = float(config['gr_eps'])

        args.optimizer = optimizer_name
        args.seed = seed

        args.stop_step = 50

        config.update(vars(args))

        config = Config(**config)

        wandb.config.update(config, allow_val_change=True)

        if task == 'function':
            from tests.test_function import test_function
            test_function(wandb.config, seed)
        elif task == 'network':
            from tests.test_network import test_network
            test_network(wandb.config, seed)
        else:
            raise NotImplementedError("Only function task is supported for now")

        wandb.finish()