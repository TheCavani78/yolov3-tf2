import tensorflow as tf
import numpy as np


def relu(x, lam=1):
    return x if x > 0 else x/lam


def selu_like(x, lam=1):
    return x if x > 0 else (1/lam)*(tf.exp(lam*x) - 1)


def selu_like2(x, lam=1):
    return x if x > 0 else (1/lam)*(tf.tanh(lam*x))


def selu_like3(x, lam=1):
    return x if x > 0 else (1/lam)*(tf.atan(lam*x))


def smooth_relu1(x, lam=1):
    return x*(1/lam + (1-1/lam) * (1 + tf.atan(lam*x)*2/np.pi)/2)


def smooth_relu2(x, lam=1):
    return x*(1/lam + (1-1/lam) * (1 + tf.atan(lam*x)*2/np.pi)/2) + (1 - 1/lam)/lam/np.pi


def smooth_relu3(x, lam=1):
    return x*(1/lam + (1-1/lam) * (1 + tf.tanh(lam*x))/2)

