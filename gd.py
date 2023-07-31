import numpy as np
from formula import mse_sigmoid_grad, mse, sigmoid
from utils import add_biases
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

matplotlib.use("tkagg")

def GD(
    w0: np.ndarray,   # init weights
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int = 1,    # default as stochastic
    eta: float = 1e-3,  # learning rate
    beta1: float = 0.9, 
    beta2: float = 0.999, 
    epsilon = 1e-8,
) -> list[np.ndarray]:
    w0, x, y = np.array(w0), np.array(x), np.array(y)
    w = w0
    N = x.shape[1]
    m = v = g = 0
    losses = [mse(w, x, y)]

    t = 1
    for epoch in range(1, epochs+1):
        indices = np.random.permutation(N)
        for (i, point) in zip(indices, range(N)):
            t += 1
            xi = x[:, i].reshape(-1, 1)
            yi = y[:, i].reshape(-1, 1)
            g = g + mse_sigmoid_grad(w, xi, yi)
            if point % batch_size == 0 or point == N:
                m = beta1*m + (1 - beta1)*g
                m = m / (1 - beta1**t)

                v = beta2*v + (1 - beta2)*g*g
                v = v / (1 - beta2**t)

                w = w - eta*m / np.sqrt(v + epsilon)
                g = 0
        
        loss = mse(w, x, y)
        print("Loss after {}th epoch: {}".format(epoch, loss))
        losses.append(loss)

        
    plt.plot(range(epochs+1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    return w