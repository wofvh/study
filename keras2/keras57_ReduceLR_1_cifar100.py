from gc import callbacks
from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
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

#2. 모델 

drop=0.2
optimizer ='adam'
activation='relu'

inputs = Input(shape=(32,32,3),name='input')
x = Conv2D(64,(2,2), padding='valid',
           activation=activation,name='hidden1')(inputs)    #27,27,128
x = Dropout(drop)(x)
# x = Conv2D(64,(2,2), padding='same',                        #27,27,64
#            activation=activation,name='hidden2')(x)
# x = Dropout(drop)(x)
x = MaxPool2D(2,2)(x)
x = Conv2D(32,(3,3), padding='valid',                       #25,25,32
           activation=activation,name='hidden3')(x)
x = Dropout(drop)(x)

# x = Flatten()(x)                                              # (None,25*25*32) =20000
x = GlobalAveragePooling2D()(x)
# flatten에 연산량이 많아진다는 문제를 해결하는 방법  / 평균으로 뽑아낸다 
x = Dense(100, activation=activation,name='hidden4')(x)
x = Dropout(drop)(x)

outputs = Dense(100, activation='softmax',name ='outputs')(x)
model= Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일
model.compile(optimizer=optimizer,metrics=['acc'],
                loss='categorical_crossentropy')
    

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

start = time.time()
model.fit(x_train,y_train, epochs=100, batch_size=128,validation_split=0.2,callbacks=[es,reduced_lr])
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

# 걸린시간 234.23668766021729
# loss 2.960489273071289
# acc 0.2685999870300293

