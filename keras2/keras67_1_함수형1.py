from keras.models import Model
from keras.layers import Dense,Flatten,Input
from keras.applications import VGG16
from keras.datasets import cifar100
import numpy as np

import tensorflow as tf
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
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,GlobalAveragePooling2D
from keras.applications import VGG16

vgg16 = VGG16(include_top=False)

# vgg16.trainable=False

input1 = Input(shape=(32,32,3))          
vgg1 = vgg16(input1)
dense2 = GlobalAveragePooling2D()(vgg1)          
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(16, activation='relu')(dense3)
output1 = Dense(100,activation='softmax')(dense4)

model = Model(inputs = input1, outputs = output1)

# model.trainable = False

model.summary()
                                            
print(len(model.weights))                            
print(len(model.trainable_weights))         

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






