
from gc import callbacks
from tabnanny import verbose
from sklearn. datasets import load_boston 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

#1. 데이터
datasets =  load_boston()
x, y  = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)

#2. 모델구성

model = Sequential()
model.add(Dense(64, activation= 'relu', input_dim=13))
model.add(Dense(32, activation= 'linear'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))        # relu 강력한놈
model.add(Dense(4, activation= 'linear'))
model.add(Dense(2, activation= 'relu'))      # linear = 기본값으로 생략 가능(회귀모델) 
model.add(Dense(1, activation= 'linear'))   


# #3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f >초              

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'k24_',date, '_', filename])
                      ) 
        
        
        
hist = model.fit(x_train, y_train, epochs =100, batch_size = 32, 
                 verbose=1, validation_split = 0.2,
                 callbacks = [earlystopping, mcp])                                    # callbacks으로 불러온다 erlystopping   

# model.save('./_save/keras24_3_save_model.h5')

#model = load_model('./_save/_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

#4 평가 예측
print('=======================================1. 기본출력')

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('r2 스코어 :', r2)
'''''
print('=======================================1. load_model 출력')
model2 =load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print("loss2 : ", loss2)

y_predict2 = model2.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict2)
print('r2 스코어 :', r2)

print('=======================================1. ModelCheckPonit 출력')
model3 =load_model('./_save/_ModelCheckPoint/keras24_ModelCheckPoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print("loss3 : ", loss3)

y_predict3 = model3.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict3)
print('r2 스코어 :', r2)

# =======================================1. 기본출력
# 5/5 [==============================] - 0s 2ms/step - loss: 111.6782
# loss :  111.67816925048828
# r2 스코어 : -0.35175610693173587
# =======================================1. load_model 출력
# 5/5 [==============================] - 0s 2ms/step - loss: 111.6782
# loss2 :  111.67816925048828
# r2 스코어 : -0.35175610693173587
# =======================================1. ModelCheckPonit 출력        
# 5/5 [==============================] - 0s 2ms/step - loss: 111.6782
# loss3 :  111.67816925048828
# r2 스코어 : -0.35175610693173587

'''