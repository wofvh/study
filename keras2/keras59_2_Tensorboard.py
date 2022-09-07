
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM #Flatten평평하게해라.  # 이미지 작업 conv2D 
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM,Dropout, Conv1D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from tensorflow.python.keras.layers import GlobalAveragePooling2D
#1. 데이터
(x_train, y_train), (x_test, y_test) =mnist.load_data()

print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28* 28*1)  # input 28,28,1 
x_test = x_test.reshape(10000, 28* 28*1)    # 

print(x_train.shape)            # (60000, 784)
print(np.unique(y_train, return_counts =True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# # scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) 
# scaler.transform(x_test)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

x_train = x_train.reshape(60000, 28, 28,1)
x_test = x_test.reshape(10000, 28, 28,1)

# y_train = pd.get_dummies((y_train))
# y_test = pd.get_dummies((y_test))

print(x_train.shape)
print(y_train.shape)
print(x_train)
print(y_train)

# 실습 acc 0.98이상 
# 원핫인코딩 

#2. 모델구성 

                                            
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5),   
                 padding='same',
                 input_shape=(28, 28, 1)))                                 
model.add(MaxPooling2D())                                               
model.add(Conv2D(32, (2,2), activation= 'relu'))                                                            
model.add(GlobalAveragePooling2D())                        
model.add(Dense(16, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation= 'softmax'))
model.summary()

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

es = EarlyStopping(monitor='val_loss',patience=15,mode='min',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=7,mode='auto',verbose=1,factor=0.5)

tb = TensorBoard(log_dir='D:\study_data/tensorboard_log/_graph',histogram_freq=0,write_graph=True,
                 write_images=True)

# 실행방법 = tensorboard --logdir=.(경로)
# (tf282gpu) D:\study_data\tensorboard_log\_graph>tensorboard --logdir=.
# http://localhost:6006/
# http://127.0.0.1:6006/


from keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)

# model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'],optimizer=optimizer)
model.compile(optimizer='adam',metrics=['acc'],loss='sparse_categorical_crossentropy')

import time
start = time.time()
hist = model.fit(x_train,y_train, epochs=100, batch_size=32, verbose=1,
                 callbacks=[es,reduced_lr,tb],validation_split=0.2)
end = time.time()- start


loss, acc = model.evaluate(x_test,y_test)
print('learning_rate:',learning_rate)
print('loss:',round(loss,4))
print('acc:',round(acc,4))
print('걸린시간: ',round(end,4))

##############시각화######################
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2 
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.',c='red',label='acc')
plt.plot(hist.history['val_acc'], marker='.',c='blue',label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc','val_acc'])

plt.show()