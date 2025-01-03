"""
find the best hyperparameters using wandb sweeps
"""
import wandb

wandb.finish()

wandb.init(sync_tensorboard=True,settings=wandb.Settings(start_method='thread'))
config = wandb.config
task = config.task
optimizer_name = config.optimizer

# warmup_config = config.copy()
# warmup_config.epochs = 1

if task == 'function':
    from tests.test_function import test_function
    test_function(config,config.seed)
elif task == 'network':
    from tests.test_network import test_network
    test_network(config,config.seed)
elif task == 'gpt':
        from tests.test_gpt import test_gpt
        test_gpt(wandb.config, config.seed)
else:
    raise NotImplementedError(f"task {task} is supported for now")

wandb.finish()
