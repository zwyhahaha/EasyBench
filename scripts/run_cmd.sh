# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1 python run.py --model logreg --task network --dataset LIBSVM_iris --epochs 50 --batch_size 16 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=1 python run.py --model logreg --task network --dataset LIBSVM_iris --epochs 50 --batch_size 64 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=2 python run.py --model logreg --task network --dataset LIBSVM_iris --epochs 50 --batch_size 128 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run.py --model logreg --task network --dataset LIBSVM_vowel --epochs 50 --batch_size 16 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run.py --model logreg --task network --dataset LIBSVM_vowel --epochs 50 --batch_size 64 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run.py --model logreg --task network --dataset LIBSVM_vowel --epochs 50 --batch_size 128 &
# wait
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 0 --seed 0 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 0 --seed 1 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 0 --seed 2 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 1e-4 --seed 0 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 1e-4 --seed 1 &
WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 1e-4 --seed 2 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=5 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 1e-2 --scheduler ExponentialLR --lr_decay 0.99 --seed 0 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=4 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 1e-2 --scheduler ExponentialLR --lr_decay 0.99 --seed 1 &
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce CUDA_VISIBLE_DEVICES=3 python run_params.py --model logreg --task network --dataset LIBSVM_mnist --epochs 10 --batch_size 128 --weight_decay 1e-2 --scheduler ExponentialLR --lr_decay 0.99 --seed 2 &
wait