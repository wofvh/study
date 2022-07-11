#칼라
#분류 
# 32

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout #Flatten평평하게해라.  # 이미지 작업 conv2D 
from keras.datasets import mnist, cifar100 , fashion_mnist
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler ,RobustScaler
#1. 데이터
(x_train, y_train), (x_test, y_test) =cifar100.load_data()

print(x_train.shape, y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)      # (10000, 32, 32, 3) (10000, 1)


x_train = x_train.reshape(50000, 32* 32* 3)       
x_test = x_test.reshape(10000, 32* 32* 3)        

print(x_train.shape)
print(np.unique(y_train, return_counts =True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = StandardScaler()
scaler.fit(x_train) 
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
print(x_train.shape)
print(x_test.shape)



# 원핫인코딩 

#2. 모델구성 

model = Sequential()
model.add(Dense(10,input_shape = (32*32*3,)))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라 
                                                                                           # kernel_size(2*2) * 바이어스(3) + 10(output)
# model.add(MaxPooling2D())

 #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
# model.add(Conv2D(64, (5,5), 
#                  padding = 'same',         # 디폴트값(안준것과 같다.) 
#                  activation= 'relu'))    # 출력(3,3,7)                                                     
# model.add(Flatten()) # (N, 63)
model = Sequential()
# model.add(Flatten()) #  해도 돌아감
model.add(Dense(1000,input_shape=(3072,),activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(1000,activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(100, activation='softmax'))
model.summary()

#3. 컴파일 구성 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=500, batch_size=1080,verbose=1,
                 validation_split=0.25, callbacks=[earlystopping])


# model.save("./_save/keras23_9_load_diabet.h5")
# model = load_model("./_save/keras23_9_load_diabet.h5")

#4. 평가, 예측\
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

# (kernel_size * channls + bias) * filters(output) = summary Param 개수

# loss :  4.60548210144043
# acc :  0.01

# loss :  7.965750217437744
# acc :  0.1304

# loss :  3.6936562061309814
# acc :  0.1375

loss :  5.922355651855469
acc :  0.2727