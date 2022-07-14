from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.datasets import load_diabetes
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
    )

scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)

print(x_train.shape,x_test.shape)  # (309, 10) (133, 10)

x_train = x_train.reshape(309, 10,1)
x_test = x_test.reshape(133, 10,1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(LSTM(350, input_shape=(10,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()

# input1 = Input(shape=(10,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(50, activation='relu')(dense1)
# dense3 = Dense(30, activation='sigmoid')(dense2)
# output1 = Dense(1)(dense3)

# model = Model(inputs = input1, outputs = output1)

import time
start_time = time.time()
# model = load_model("./_save/keras23_9_load_diabet.h5")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/3diabetes/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=200, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'3diabetes_',date, '_', filename])
                      ) 

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, 
#                               restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=500, batch_size=32,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping, mcp])

end_time = time.time() - start_time

# model.save("./_save/keras23_9_load_diabet.h5")
# model = load_model("./_save/keras23_9_load_diabet.h5")

#4. 평가, 예측\
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print("걸린시간 : ", end_time)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

#1.  하기전 
# 걸린시간 :  1.2999696731567383
# loss :  [28918.599609375, 150.64321899414062]
# r2스코어 :  -3.6415536423300496

# 걸린시간 :  0.7596209049224854
# loss :  [28918.599609375, 150.64321899414062]
# r2스코어 :  -3.6415536423300496

# dropout 
# 걸린시간 :  20.8634614944458
# loss :  3090.599853515625
# r2스코어 :  0.5039460632828794

# # LSTM 
# 걸린시간 :  379.0210268497467
# loss :  4140.86669921875
# r2스코어 :  0.3353738929896172
