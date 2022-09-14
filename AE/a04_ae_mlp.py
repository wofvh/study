

import numpy as np
from keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size,input_shape= (784,),activation='relu'))
    model.add(Dense(units=784,activation='sigmoid'))
    return model

def autoencoder2(hidden_layer_size1,hidden_layer_size2,hidden_layer_size3,hidden_layer_size4,hidden_layer_size5):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size1,input_shape= (784,),activation='relu'))
    model.add(Dense(units=hidden_layer_size2,activation='sigmoid'))
    model.add(Dense(units=hidden_layer_size3,activation='sigmoid'))
    model.add(Dense(units=hidden_layer_size4,activation='sigmoid'))
    model.add(Dense(units=hidden_layer_size5,activation='sigmoid'))
    model.add(Dense(units=784,activation='sigmoid'))
    return model

# model= autoencoder(hidden_layer_size=154) # pca의 95% 성능
model= autoencoder(hidden_layer_size=331) # pca의 99% 성능
model2 = autoencoder2(100,200,300,400,500)

model.compile(optimizer='adam',loss='mse')
model.fit(x_train,x_train,epochs=10)
output = model.predict(x_test)

model2.compile(optimizer='adam', loss='mse')
model2.fit(x_train, x_train, epochs=10)
output2 = model2.predict(x_test)

from matplotlib import pyplot as plt
import random
fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15))= \
    plt.subplots(3,5,figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지 맨위에 그린다.
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0 :
        ax.set_ylabel('input',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
# 오토인콛더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0 :
        ax.set_ylabel('output',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0 :
        ax.set_ylabel('output',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()

    