

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from keras.datasets import mnist, cifar10 , fashion_mnist
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
(x_train, y_train), (x_test, y_test) =cifar10.load_data()

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

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(x_train.shape)
# print(x_test.shape, x_train.shape)      # (10000, 32, 32, 3) (50000, 32, 32, 3)
print(x_train.shape)    # (50000, 32, 32, 3)
print(y_train.shape)    # (50000, 10)
print(x_train.shape,x_test.shape) # (50000, 3072) (10000, 3072)


x_train = x_train.reshape(50000, 32,96)
x_test = x_test.reshape(10000, 32,96)


# 원핫인코딩 

#2. 모델구성 

                                    
model = Sequential()    
model.add(Conv1D(1280, 2, input_shape=(32,96)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(10, activation='softmax')) 


#3. 컴파일 구성 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='min', 
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=200, batch_size=1000,verbose=2,
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
print('conv1D cifar10')

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


# loss :  4.3395161628723145
# acc :  0.519

# loss :  3.9987804889678955
# acc :  0.600
# 함수형
# loss :  1.5904154777526855
# acc :  0.4591


# LSTM
#  accuracy: 0.0749
# loss :  5.423413276672363
# acc :  0.0749

# Conv1D 
