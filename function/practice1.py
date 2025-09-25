import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from keras import Input
# from keras.models import Model
# from keras.layers import Dense
import matplotlib.pyplot as plt

def test_system(x):
    return 0.4*x+0.8

if __name__ == "__main__":
    x_datas = np.array(range(-50, 51, 10))
    y_datas = []
    for x in x_datas:
        y_datas.append(test_system(x))
    y_datas = np.array(y_datas)

    plt.scatter(x_datas, y_datas)
    plt.show()