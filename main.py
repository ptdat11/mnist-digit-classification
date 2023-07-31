import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from formula import mse_sigmoid_grad, sigmoid
from gd import GD
from utils import one_hot, add_biases

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).T
x_train = add_biases(x_train, 0)
y_train = one_hot(y_train, 10)

def test(w):
    correct = 0
    for i in range(x_test.shape[1]):
        out = w @ x_test[:, i]
        pred = out.argmax()
        label = y_test[i]
        correct += pred == label
    print("Reliability: {}%".format(correct / y_test.shape[0] * 100))

w = GD(
    w0=np.load("weights.npz", "r")["arr_0"],
    x=x_train,
    y=y_train,
    epochs=1000,
    eta=1e-7,
    batch_size=3000,
    beta2=0.99,
) 
np.savez("weights.npz", w)

x_test = x_test.reshape(-1, 784).T
x_test = add_biases(x_test, 0)

correct = 0
for i in range(x_test.shape[1]):
    out = w @ x_test[:, i]
    pred = out.argmax()
    label = y_test[i]
    correct += pred == label
print("Reliability: {}%".format(correct / y_test.shape[0] * 100))