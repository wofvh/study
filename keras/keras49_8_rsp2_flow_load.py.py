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
x_train = np.load('d:/study_data/_save/_npy/keras49_8_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_8_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_8_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_8_test_y.npy')

print(x_train.shape)            # (40000, 100, 100, 3)
print(y_train.shape)            # (40000,)
print(x_test.shape)             # (504, 100, 100, 3)
print(y_test.shape)             # (504,)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D

#2. 모델 
model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (100,100,3),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(3,activation='softmax'))
    

#3. 컴파일.

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1])          # 배치를 최대로 잡으면 가능
hist = model.fit(x_train,y_train, epochs=5,validation_split=0.3,verbose=2,batch_size=100) 
                    # steps_per_epoch=32,  # steps_per_epoch=32 데이터를 batch size로 나눈것. 160/5 =32 
                    # validation_data=xy_test,
                    # validation_steps=4)

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


# loss :  9.834131731167872e-08
# val_loss :  0.03201524540781975
# accuracy :  1.0
# val_accuracy :  0.9966942071914673

# loss :  1.0985032320022583
# val_loss :  1.0987937450408936
# accuracy :  0.3424285650253296
# val_accuracy :  0.3333333432674408
# predict :  [0.24719454 0.05714595 0.6956595 ]