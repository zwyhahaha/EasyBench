#!/bin/bash
export WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 11 --batch_size 128 --weight_decay 0 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=2 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 11 --batch_size 128 --weight_decay 0 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 11 --batch_size 128 --weight_decay 0 --seed 2 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=2 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 11 --batch_size 128 --weight_decay 1e-4 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=6 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 11 --batch_size 128 --weight_decay 1e-4 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=7 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 11 --batch_size 128 --weight_decay 1e-4 --seed 2 &

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1 python run_params.py --optimizer OSMM2 --model gpt --task gpt --epochs 150 --weight_decay 0 --seed 1

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_letter --epochs 11 --batch_size 64 --weight_decay 0 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_letter --epochs 11 --batch_size 64 --weight_decay 0 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_letter --epochs 11 --batch_size 64 --weight_decay 0 --seed 2 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_letter --epochs 11 --batch_size 64 --weight_decay 1e-4 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_letter --epochs 11 --batch_size 64 --weight_decay 1e-4 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_letter --epochs 11 --batch_size 64 --weight_decay 1e-4 --seed 2 &

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_satimage --epochs 20 --batch_size 16 --weight_decay 1e-2 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_satimage --epochs 20 --batch_size 16 --weight_decay 1e-2 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_satimage --epochs 20 --batch_size 16 --weight_decay 1e-2 --seed 2 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_segment --epochs 20 --batch_size 16 --weight_decay 1e-2 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_segment --epochs 20 --batch_size 16 --weight_decay 1e-2 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_segment --epochs 20 --batch_size 16 --weight_decay 1e-2 --seed 2 &

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model mlp --task network --dataset MNIST --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1 python run_params.py --model mlp --task network --dataset MNIST --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=2 python run_params.py --model mlp --task network --dataset MNIST --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 2 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model mlp --task network --dataset MNIST --epochs 50 --batch_size 128 --weight_decay 0 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model mlp --task network --dataset MNIST --epochs 50 --batch_size 128 --weight_decay 0 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=7 python run_params.py --model mlp --task network --dataset MNIST --epochs 50 --batch_size 128 --weight_decay 0 --seed 2 &

# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=2 python run_params.py --model mlp --task network --dataset MNIST --epochs 10 --batch_size 128 --weight_decay 1e-4 --seed 0 &
CUDA_VISIBLE_DEVICES=5 python run_params.py --optimizer SGD --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 0 &
CUDA_VISIBLE_DEVICES=4 python run_params.py --optimizer NAG --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 0 &
CUDA_VISIBLE_DEVICES=3 python run_params.py --optimizer Adam --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 0 &
CUDA_VISIBLE_DEVICES=2 python run_params.py --optimizer OSMM --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 0 &
CUDA_VISIBLE_DEVICES=1 python run_params.py --optimizer OSMM2 --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 0 &

CUDA_VISIBLE_DEVICES=6 python run_params.py --optimizer SGD --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 1 &
CUDA_VISIBLE_DEVICES=7 python run_params.py --optimizer NAG --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 1 &
CUDA_VISIBLE_DEVICES=0 python run_params.py --optimizer Adam --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 1 &
CUDA_VISIBLE_DEVICES=2 python run_params.py --optimizer OSMM --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 1 &
CUDA_VISIBLE_DEVICES=1 python run_params.py --optimizer OSMM2 --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 1 &

CUDA_VISIBLE_DEVICES=3 python run_params.py --optimizer SGD --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 2 &
CUDA_VISIBLE_DEVICES=4 python run_params.py --optimizer NAG --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 2 &
CUDA_VISIBLE_DEVICES=5 python run_params.py --optimizer Adam --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 2 &
CUDA_VISIBLE_DEVICES=6 python run_params.py --optimizer OSMM --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 2 &
CUDA_VISIBLE_DEVICES=7 python run_params.py --optimizer OSMM2 --model resnet --task network --dataset CIFAR10 --epochs 50 --batch_size 128 --weight_decay 1e-4 --seed 2 &
wait