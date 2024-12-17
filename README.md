This repo builds a benchmark environment for optimizers.

The testing environments include:
- function: rosenbrock, rastrigin, least_squares (quadratic)
- network: logistic + MNIST, mlp + MNIST, vgg + CIFAR10
- llm: nanoGPT + shakespeare, nanoGPT + openwebtext

The workflow is composed by
- `models`: defines network structure
- `optimizers`: self-defined optimizers
- `data`: includes libsvm, mnist, cifar10
- `tests`: train models, using optimizers, on data
- `sweep.py`: for each optimizer, sweep hyperparameters like lr, on `tests`, the sweep configuration is in `scripts`
- `run.py`: run test env, on all optimizers, using their best params in `params`

How to run wandb sweep:
```
wandb sweep scripts/network_sweep.yaml
```
