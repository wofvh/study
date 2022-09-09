# trainable = True, False 비교해가면서 만들어서 결과비교 

from gc import callbacks
from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D

print(tf.__version__)


#1. 데이터

(x_train,y_train),(x_test,y_test) = cifar100.load_data()

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

import numpy as np
from sklearn.cluster import k_means
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten
from keras.applications import VGG16

vgg16 = VGG16(weights='imagenet',include_top=False,
              input_shape=(32,32,3))

vgg16.trainable=False
# vgg16.summary()

model =Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100,activation='softmax'))

# model.trainable = False

model.summary()
                                            # trainable: True / VGG False / model False
print(len(model.weights))                   # 30                / 30            / 30         
print(len(model.trainable_weights))         # 30                / 4             /  0

print(model.layers)

import pandas as pd
pd.set_option('max_colwidth',-1)
layers =[(layer,layer.name,layer.trainable)for layer in model.layers]
results = pd.DataFrame(layers,columns=['Layer Type','Layer Name','Layer Trainable'])
print(results)

model.compile(optimizer='adam',metrics=['acc'],
                loss='categorical_crossentropy')
    

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

start = time.time()
model.fit(x_train,y_train, epochs=10, batch_size=256,validation_split=0.2,callbacks=[es,reduced_lr])
end = time.time()-start

loss,acc = model.evaluate(x_test,y_test)

# print('model.score:',model.score)
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
# y_predict = np.argmax(model.predict(x_test),axis=1)
# y_test =np.argmax(y_test)
print('걸린시간',end)
print('loss',loss)
print('acc',acc)

# true
# 걸린시간 103.6434736251831
# loss 2.4836130142211914
# acc 0.359499990940094

# vgg16.trainable=False
# 걸린시간 43.168771743774414
# loss 2.63523268699646
# acc 0.3431999981403351

# model.trainable = False
# 걸린시간 42.45200252532959
# loss 4.895547866821289
# acc 0.014299999922513962




