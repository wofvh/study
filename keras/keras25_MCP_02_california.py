
from gc import callbacks
from tabnanny import verbose
from sklearn. datasets import   fetch_california_housing  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn import datasets
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

#1. 데이터
datasets =  fetch_california_housing ()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
    )
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)

#2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(120))
# model.add(Dense(80))
# model.add(Dense(25))
# model.add(Dense(1))

input1 = Input(shape=(8,))          # 컬럼3개를 받아드린다.
dense1 = Dense(64)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)


import time

# #3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/7digit/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'7digit_',date, '_', filename])
                      ) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping =EarlyStopping(monitor='loss', patience=100, mode='auto', 
              verbose=1, restore_best_weights = True)     
   

hist = model.fit(x_train, y_train, epochs =500, batch_size = 32, 
                 verbose=1, validation_split = 0.2,
                 callbacks = [earlystopping, mcp])                    # callbacks으로 불러온다 erlystopping   

# model.save("./_save/keras23_6_load_weights2.h5")



# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)


#model.save("./_save/keras23_6_load_weights3.h5")
model = load_model("./_save/keras23_6_load_weights3.h5")


#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('r2 스코어 :', r2)



