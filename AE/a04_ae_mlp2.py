

import numpy as np
from tensorflow.keras.datasets import mnist


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255


# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder1(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


def autoencoder2(hidden_layer_size1, hidden_layer_size2, hidden_layer_size3, hidden_layer_size4, hidden_layer_size5):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size1, input_shape=(784,), activation='relu'))
    model.add(Dense(units=hidden_layer_size2, activation='relu'))
    model.add(Dense(units=hidden_layer_size3, activation='relu'))
    model.add(Dense(units=hidden_layer_size4, activation='relu'))
    model.add(Dense(units=hidden_layer_size5, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model1 = autoencoder1(hidden_layer_size=154)      # pca 95% : 154
model1.compile(optimizer='adam', loss='mse')
model1.fit(x_train, x_train, epochs=10)
output1 = model1.predict(x_test)


model2 = autoencoder2(404,202,50,202,404)
model2.compile(optimizer='adam', loss='mse')
model2.fit(x_train, x_train, epochs=10)
output2 = model2.predict(x_test)


from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(30, 7))


# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output1.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 오토인코더1이 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 오토인코더2이 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output2[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()