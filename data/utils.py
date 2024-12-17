import torch

def generate_least_squares_data(num_samples=1000, num_features=3, noise=0.1, seed=42, lib=torch):
    lib.manual_seed(seed)
    X = lib.randn(num_samples, num_features)
    true_w = lib.randn(num_features, 1)
    y = X @ true_w + noise * lib.randn(num_samples, 1)
    return X, y