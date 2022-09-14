import numpy as np
from keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()

# input과 output이 똑같다.
# 필요없는것을 버리고 특성만 뽑아서 재조성하는것. 

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)     
# encoded = Dense(1064, activation='relu')(input_img)   # 노드를 늘릴경우 (정확히표현)
# encoded = Dense(8, activation='relu')(input_img)       # 노드를 줄일경우 (형태유지)

# decoded = Dense(784, activation='sigmoid')(encoded)    # 최초
# decoded = Dense(784, activation='linear')(encoded)     # 최악       
# decoded = Dense(784, activation='relu')(encoded)       # 별로       
decoded = Dense(784, activation='sigmoid')(encoded)       # 최악       


autoencoder = Model(input_img,decoded)
autoencoder.summary()#

autoencoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

autoencoder.fit(x_train,x_train,epochs=30,batch_size=128,validation_split=0.2)
 
decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10 
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
plt.show()


