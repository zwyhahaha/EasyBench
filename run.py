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
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for training')
parser.add_argument('--batches', type=int, required=True, help='Number of batches for training')
parser.add_argument('--input_dim', type=int, required=True, help='Input dimension for the model')

args = parser.parse_args()

model = args.model
task = args.task
epochs = args.epochs
batches = args.batches
input_dim = args.input_dim
optimizers = ['SGD', 'NAG', 'Adam', 'OSGM', 'OSMM']

wandb.init(project=f'run_{model}_{task}_{epochs}_{batches}_{input_dim}')

for optimizer_name in optimizers:
    config_path = f'params/{model}/{task}/{optimizer_name}.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    wandb.config.update(config)

    wandb.config.model = model
    wandb.config.task = task
    wandb.config.epochs = epochs
    wandb.config.batches = batches
    wandb.config.input_dim = input_dim

    if task == 'function':
        from tests.test_function import test_function
        test_function(wandb.config)
    else:
        raise NotImplementedError("Only function task is supported for now")

wandb.finish()