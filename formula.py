import numpy as np
from typing import Sequence, Callable

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
    Calling f(x) an activation function
    MSE(W): R^(mxn) -> R
        = 1/2N * ||y - f(Wx)||_2 ^2
'''
def mse(
    w: Sequence, # R^(mxn)
    x: Sequence, # R^n
    y: Sequence,  # R^m
    activation_func: Callable = None
) -> float:
    w, x, y = np.array(w), np.array(x), np.array(y)
    N = x.shape[1]
    if activation_func is not None:
        errors = y - activation_func(w @ x)
    else: 
        errors = y - w @ x
    return np.sum(errors*errors) / 2 / N


'''
    grad(W): R^(mxn) -> R^(mxn)
        = ∇ MSE(W) 
        = ∇ (1/2 * ||y - f(Wx)||_2 ^2)
        = (y - f(Wx)) (∇f(Wx)) x
    Single point gradient doesn't divide by N
'''
def mse_sigmoid_grad(
    w: Sequence, # R^(mxn)
    x: Sequence, # R^n
    y: Sequence  # R^m 
) -> np.ndarray:
    w, x, y = np.array(w), np.array(x), np.array(y)
    grad = (w @ x - y) @ x.T
    return grad