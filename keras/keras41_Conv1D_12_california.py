

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
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM, Conv1D, Flatten

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

print(x_train.shape,x_test.shape)  # (14447, 8) (6193, 8)


x_train = x_train.reshape(14447, 8,1)
x_test = x_test.reshape(6193, 8,1)

#2. 모델구성
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(Conv1D(128, 2, input_shape=(8,1)))
model.add(Flatten())
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()


# input1 = Input(shape=(8,))                            # 컬럼3개를 받아드린다.
# dense1 = Dense(64)(input1)                            # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(32, activation='relu')(dense1)
# drop1 = Dropout(0.2)(dense2)
# dense3 = Dense(16, activation='sigmoid')(drop1)
# drop2 = Dropout(0.3)(dense3)
# output1 = Dense(1)(dense3)(drop2)
# model = Model(inputs = input1, outputs = output1)




# #3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')           # 0707_1723
# print(date)

# filepath = './_ModelCheckPoint/7digit/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
#                       save_best_only=True,                                      # patience 필요없음.
#                       filepath ="".join([filepath,'7digit_',date, '_', filename])
#                       ) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping =EarlyStopping(monitor='loss', patience=100, mode='auto', 
              verbose=1, restore_best_weights = True)     

import time
start_time = time.time()

hist = model.fit(x_train, y_train, epochs =500, batch_size = 4000, 
                 verbose=1, validation_split = 0.2,
                 callbacks = [earlystopping]) 
end_time = time.time() - start_time  

# model.save("./_save/keras23_6_load_weights2.h5")



# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)


#model.save("./_save/keras23_6_load_weights3.h5")
# model = load_model("./_save/keras23_6_load_weights3.h5")


#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('r2 스코어 :', r2)
print('시간 :',end_time)

# LSTM
# loss :  0.3155742883682251
# r2 스코어 : 0.7700181355774813

# Conv1D 
# loss :  0.2961958646774292
# r2 스코어 : 0.7841406486562685
# 시간 : 27.860892057418823