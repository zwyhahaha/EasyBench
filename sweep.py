"""
find the best hyperparameters using wandb sweeps
"""
import wandb

wandb.finish()

wandb.init(sync_tensorboard=True,settings=wandb.Settings(start_method='thread'))
config = wandb.config
task = config.task
optimizer_name = config.optimizer

assert task in ['function', 'network', 'llm']
assert optimizer_name in ['SGD', 'NAG', 'Adam', 'OSGM', 'OSMM', 'OSMM2']

if task == 'function':
    from tests.test_function import test_function
    test_function(config,config.seed)
elif task == 'network':
    from tests.test_network import test_network
    test_network(config,config.seed)
else:
    raise NotImplementedError("Only function task is supported for now")

wandb.finish()
