# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/vowel_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/iris_all_optimizers.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/mnist_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/iris_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/vowel_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/vehicle_osmm_tune.yaml
# WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb sweep scripts/network/logreg/letter_osmm_tune.yaml
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$(i) WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce  wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_vowel_osmm_dampening/0lewrbhd &
    # CUDA_VISIBLE_DEVICES=$(i) WANDB_API_KEY=d2d00dff74b3ad422b1b715587ed1a2089c640ce wandb agent wanyuzhang1013-shanghai-university-of-finance-and-economics/network_logreg_mnist_osmm_tune/75bz9sp0 &
done
wait
