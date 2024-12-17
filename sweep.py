"""
find the best hyperparameters using wandb sweeps
"""
import wandb

wandb.init()
config = wandb.config
task = config.task # ['function', 'network', 'llm']
optimizer_name = config.optimizer # 'SGD', 'NAG', 'Adam', 'OSGM', 'OSMM'

assert task in ['function', 'network', 'llm']
assert optimizer_name in ['SGD', 'NAG', 'Adam', 'OSGM', 'OSMM']

if task == 'function':
    from tests.test_function import test_function
    test_function(config)
elif task == 'network':
    from tests.test_network import test_network
    test_network(config)
else:
    raise NotImplementedError("Only function task is supported for now")

wandb.finish()
