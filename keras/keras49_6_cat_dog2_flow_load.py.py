# 넘파이에서 불러와서 모델구성
# 성능비교
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D #Flatten평평하게해라.  # 이미지 작업 conv2D 

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 


#1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_6_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_6_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_6_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_6_test_y.npy')

print(x_train.shape)            # (40000, 150, 150, 1)
print(y_train.shape)            # (40000,)
print(x_test.shape)             # (40000, 150, 150, 1)
print(y_test.shape)             # (10000, )


# x_train = x_train.reshape(40000, 150, 150, 1)
# x_test = x_test.reshape(40000, 150, 150, 1)
# print(x_train)
# print(x_test)
#2 모델구성 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (150,150,1),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
    

#3. 컴파일

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])
hist = model.fit(x_train, y_train, epochs=5, batch_size=5,verbose=2,
                 validation_split=0.3)
                  

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-1])
print('val_loss : ',val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('predict : ',y_predict[-1])

# loss :  2.0596190565221918e-14
# val_loss :  24.834657669067383
# accuracy :  1.0
# val_accuracy :  0.5
