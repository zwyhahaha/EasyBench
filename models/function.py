"""
This file defines both convex and nonconvex functions for testing optimization algorithms.
"""
import math
import torch
import numpy as np


def rosenbrock(tensor):
    """
    Nonconvex Rosenbrock function, optimal value at f(1, 1, ..., 1) = 0
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    if not isinstance(tensor, (list, np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a list, numpy array, or torch tensor")
    if len(tensor) < 2:
        raise ValueError("Input must have at least two elements")
    return sum(100 * (tensor[i+1] - tensor[i]**2)**2 + (1 - tensor[i])**2 for i in range(len(tensor) - 1))

def rastrigin(tensor, lib=torch):
    """
    Nonconvex Rastrigin function, optimal value at f(0, 0, ..., 0) = 0
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    if not isinstance(tensor, (list, np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a list, numpy array, or torch tensor")
    if len(tensor) == 0:
        raise ValueError("Input must not be empty")
    A = 10
    n = len(tensor)
    return A * n + sum((x**2 - A * lib.cos(2 * math.pi * x)) for x in tensor)

def least_squares(tensor, X, y, lib=torch):
    """
    Convex least squares function, or quadratic function, optimal value 0
    """
    if not isinstance(tensor, (np.ndarray, torch.Tensor)):
        raise TypeError("Input tensor must be a numpy array or torch tensor")
    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise TypeError("Input X must be a numpy array or torch tensor")
    if not isinstance(y, (np.ndarray, torch.Tensor)):
        raise TypeError("Input y must be a numpy array or torch tensor")
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of rows in X must be equal to the number of elements in y")
    if X.shape[1] != tensor.shape[0]:
        raise ValueError("The number of columns in X must be equal to the number of elements in tensor")
    
    loss = lib.mean((y - X @ tensor) ** 2) / 2
    return loss
