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
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
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

scaler = StandardScaler()
scaler.fit(x_train) 
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
print(x_train.shape)
print(x_test.shape)



# 원핫인코딩 

#2. 모델구성 

model = Sequential()
model.add(Conv2D(64, kernel_size=(6, 6),   # 출력(4,4,10)                                       # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(32, 32, 3), activation= 'softmax'))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라                                                                        
model.add(Conv2D(64, (5,5), 
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'relu'))    # 출력(3,3,7)                                                     
model.add(Flatten()) # (N, 63)
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation= 'softmax'))
model.summary()
# model.add(Dense(1000,input_shape=(3072,),activation='swish'))
# model.add(Dropout(0.3))
# model.add(Dense(1000,activation='swish'))
# model.add(Dropout(0.3))
# model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(100, activation='softmax'))

#3. 컴파일 구성 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='min', 
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=100, batch_size=1080,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping])


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

# tf.keras.layers.Dense(
#     units,                                # output 로드 개수 10 
#     activation=None,
#     use_bias=True,                        # 
#     kernel_initializer="glorot_uniform",  # 레이어 초기화
#     bias_initializer="zeros",             # 레이어 초기화
#     kernel_regularizer=None,              # 정규화, 규제화 
#     bias_regularizer=None,                # 정규화, 규제화 
#     activity_regularizer=None,            # 정규화, 규제화 
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs)

#      model.add(Dense(10, activation ='relu', input_dim =8)
#      2차원일때 input shape ) Dense > (batch_size(행),input_dim(열))



# loss :  4.60548210144043
# acc :  0.01

# loss :  7.965750217437744
# acc :  0.1304

# loss :  6.895402908325195
# acc :  0.2805