# VGG16,VGG19,ResNet50,ResNet152V2,ResNet101, ResNet101V2, ResNet152,  InceptionV3, InceptionResNetV2, DenseNet169,
#         MobileNet, MobileNetV2,MobileNetV3Small,MobileNetV3Large,NASNetLarge,NASNetMobile,EfficientNetB0,EfficientNetB1,EfficientNetB7,Xception]
        
# VGG16 
# VGG19     
# Xception
# ResNet50
# ResNet101
# InceptionV3
# InceptionResNetV2
# Dense121
# MobileNetV2
# NasNetMobile
# EfficeintNetB0

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

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
from keras.applications import VGG16, VGG19

models =VGG19(weights='imagenet',include_top=False,
              input_shape=(32,32,3))

models.trainable=False
# vgg16.summary()

model =Sequential()
model.add(models)
model.add(Flatten())
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
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))

print('loss: ', loss)
print('acc: ', acc)

# 걸린시간 50.16588020324707
# loss 2.7248916625976562
# acc 0.3190000057220459









