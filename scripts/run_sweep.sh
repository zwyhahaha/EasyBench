#!/bin/bash

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/vowel_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/iris_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/mnist_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/iris_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/vowel_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/letter_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/letter_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/middle_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/middle_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/mlp/all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/mlp/osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/middle_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/vgg/all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/resnet/all_optimizers.yaml

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_vgg_all_optimizers/xymtktdn
# bash scripts/run_sweep.sh

export WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce

LOG_DIR="/home/shanshu/opt/EasyBench/logs"
mkdir -p $LOG_DIR

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_vgg_all_optimizers/8ziv1rst > $LOG_DIR/task_$i.log 2>&1 &
done

wait