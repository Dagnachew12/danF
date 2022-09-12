# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
from dan import dand


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    x = torch.tensor([1,2,3,4,5,6])
    y = dand.tan(x)
    plt.plot(x,y)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
