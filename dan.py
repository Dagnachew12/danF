import torch
import numpy as np
import matplotlib.pyplot as plt
class dand:
    def __init__(self, x):
        super.__init__()
        self.var = x

    def relu(x):
        return max(0.0, x)

    def leak_relu(x, a):
        if x > 0:
            return x
        else:
            return a * x

    def sigmoid(x):
        return 1 / (1 + np.exp(-x));

    def tan(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def gradientSigmoid(x):
        return x.sigmoid() * (1 - x.sigmoid())

    def derivative_tanh(x):
        return 1 - x.tan() * x.tan()

    def softmax(x):
        x_exp = torch.exp(x)
        partition = x_exp.sum(1, keepdim=True)
        return x_exp / partition



