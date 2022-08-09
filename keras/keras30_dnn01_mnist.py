
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from tensorflow.keras.datasets import mnist
import numpy as np
# from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
(x_train, y_train), (x_test, y_test) =mnist.load_data()

print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28* 28* 1)  # input 28,28,1 
x_test = x_test.reshape(10000, 28* 28* 1)    # 

print(x_train.shape)
print(np.unique(y_train, return_counts =True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train) 
# # scaler.transform(x_test)
# x_test =scaler.transform(x_test)
# x_train = scaler.transform(x_train)
# # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
# #       dtype=int64))

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

y_train = pd.get_dummies((y_train))
y_test = pd.get_dummies((y_test))

print(x_train.shape)    # (60000, 784)
print(y_train.shape)    # (60000, 10)
# 실습 acc 0.98이상 
# 원핫인코딩 

#2. 모델구성 
model = Sequential()
# model.add(Dense(64, input_shape =(28*28, )))
# # model.add(Dense(64, input_shape =(784,)))
# # model = Sequential()
model.add(Dense(units=10, input_shape=(28 * 28,)))   # 출력(4,4,10)          # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                
                   #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라 
                                            # kernel_size(2*2) * 바이어스(3) + 10(output)
# model.add(Deooling2D())                   # MaxPooling2D 위에 붙은것과 같고, 자원낭비가 조금 있다. 가장큰값만 반환한다. () 안에 값을 넣을 수 있다.         
#                                             #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
# model.add(Conv2D(32, (2,2), 
#                  padding = 'valid',         # 디폴트값(안준것과 같다.) 
#                  activation= 'relu'))       # 출력(3,3,7)                                                     
model.add(Flatten())                        # (N, 63) 위치와 순서가 변경되면 안된다. transpose는 조작으로 Flatten과 헷갈리기 쉬우니 조심. 
model.add(Dense(100, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일 구성 
import time
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=30, mode='auto', 
              verbose=1, restore_best_weights = True)     

start = time.time()
        
hist = model.fit(x_train, y_train, epochs=300, batch_size=320,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping])

end = time.time() - start


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
print("걸린시간 : ", end)

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

# oss: 0.2821 - accuracy: 0.9779
# loss :  0.28206199407577515
# acc :  0.9779

# loss :  0.12605245411396027
# acc :  0.9853

# dnn
# loss :  0.3424288332462311
# acc :  0.922


