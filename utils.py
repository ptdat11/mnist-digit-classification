import numpy as np
from typing import Sequence

def one_hot(data: Sequence[int], max: int = None) -> np.ndarray:
    darr = np.array(data)
    size = darr.size
    _max = max if max is not None else darr.max()+1
    frame = np.zeros((size, _max))
    frame[np.arange(size), darr] = 1
    return frame.T

def add_biases(
    a: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    a = np.array(a)
    ones_dim = list(a.shape)
    ones_dim[axis] = 1
    ones = np.ones(ones_dim)
    return np.concatenate([
            ones,
            a
        ], axis=axis)