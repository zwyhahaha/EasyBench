# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/mlp/all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/iris_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_osgm_tune/j0zwgf7z
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1  wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_iris_all_optimizers/y6n81ju3 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=2  wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_iris_all_optimizers/y6n81ju3 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3  wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_iris_all_optimizers/y6n81ju3 &
wait
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce python run.py --model logreg --task network --dataset MNIST --epochs 20 --batch_size 128
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce  wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_all_optimizers/aqu13qt8