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
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

y_train = pd.get_dummies((y_train))
y_test = pd.get_dummies((y_test))

print(x_train.shape)
print(y_train.shape)
print(x_train)
print(y_train)

# 실습 acc 0.98이상 
# 원핫인코딩 

#2. 모델구성 

model = Sequential()
model.add(LSTM(50, input_shape=(28, 28)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(4, activation='swish'))
model.add(Dense(10))
model.summary()


#3. 컴파일, 훈련
# from keras.optimizers import Adam,Adadelta,Adagrad,Adamax,RMSprop,SGD,Nadam
from tensorflow.python.keras.optimizer_v2 import adam, adadelta,adagrad,adamax,rmsprop,nadam

learning_rate = 0.001

# optimizer1 = adam.Adam(lr=learning_rate)            # loss: 2.5764 lr: 0.001 결과: [[11.669696]]
# optimizer2 = adadelta.Adadelta(lr=learning_rate )    # loss: 2.9061 lr: 0.001 결과: [[10.365168]]
# optimizer3 = adagrad.Adagrad(lr=learning_rate )      # loss: 2.4779 lr: 0.001 결과: [[10.500922]]
# optimizer4 = adamax.Adamax(lr=learning_rate )        # loss: 2.4894 lr: 0.001 결과: [[10.227734]]
# optimizer5 = rmsprop.RMSprop(lr=learning_rate )      # loss: 6.4987 lr: 0.001 결과: [[7.341303]]
# optimizer6 = nadam.Nadam(lr=learning_rate )          # loss: 2.9608 lr: 0.001 결과: [[8.920817]]

optimizers = [adam.Adam(lr=learning_rate) ,adadelta.Adadelta(lr=learning_rate ),adagrad.Adagrad(lr=learning_rate ),
              adamax.Adamax(lr=learning_rate) ,rmsprop.RMSprop(lr=learning_rate ) ,nadam.Nadam(lr=learning_rate ) ]
aa = []
for i in optimizers :
    model.compile(loss='categorical_crossentropy',optimizer = i, metrics=['accuracy'])
    earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True) 
    
    model.fit(x_train,y_train,epochs=20,batch_size=50,verbose=1, validation_split=0.2,
              callbacks=[earlystopping])
    
    results = model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    y_predict = tf.argmax(y_predict,axis=1) 
    
    y_test = tf.argmax(y_test,axis=1) 
    acc = accuracy_score(y_test,y_predict)
    
    
    print('results:',results,i,':','acc:',acc)
    aa.append(acc)
    print(aa)
    
exit()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True)     

import time 
start = time.time()
        
hist = model.fit(x_train, y_train, epochs=200, batch_size=300,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping])
end =  time.time()- start

results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)