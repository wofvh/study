from gc import callbacks
from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout,LSTM, Conv1D
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
print(tf.__version__)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine,load_digits
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf


from sklearn.svm import LinearSVC

#1. 데이터
datasets = load_digits()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
print(x_train.shape)
print(x_test.shape)
# (1437, 64)
# (360, 64)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(1437, 64,1)
x_test = x_test.reshape(360,64,1)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 

# drop=0.2
# optimizer ='adam'
# activation='relu'

# inputs = Input(shape=(4,1,1),name='input')
# x = Conv2D(64,(2,2), padding='valid',
#            activation=activation,name='hidden1')(inputs)    #27,27,128
# x = Dropout(drop)(x)
# # x = Conv2D(64,(2,2), padding='same',                        #27,27,64
# #            activation=activation,name='hidden2')(x)
# # x = Dropout(drop)(x)
# x = MaxPool2D(2,2)(x)
# x = Conv2D(32,(3,3), padding='valid',                       #25,25,32
#            activation=activation,name='hidden3')(x)
# x = Dropout(drop)(x)

# # x = Flatten()(x)                                              # (None,25*25*32) =20000
# x = GlobalAveragePooling2D()(x)
# # flatten에 연산량이 많아진다는 문제를 해결하는 방법  / 평균으로 뽑아낸다 
# x = Dense(1, activation=activation,name='hidden4')(x)
# x = Dropout(drop)(x)

# outputs = Dense(3, activation='softmax',name ='outputs')(x)
# model= Model(inputs=inputs, outputs=outputs)
# model.summary()
#################################################
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(Conv1D(128, 2, input_shape=(64,1)))
model.add(Flatten())
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
# from tensorflow.keras.optimizers import Adam
# learing_rate = 0.01
# optimizer = Adam(learing_rate=learing_rate)

optimizer='adam'
model.compile(optimizer=optimizer,metrics=['acc'],
                loss='categorical_crossentropy')
    

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=50,mode='auto',verbose=1,factor=0.5)

start = time.time()
model.fit(x_train,y_train, epochs=600, batch_size=128,validation_split=0.2,callbacks=[es,reduced_lr])
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

# 걸린시간 15.781590700149536
# loss 0.11197677254676819
# acc 1.0

