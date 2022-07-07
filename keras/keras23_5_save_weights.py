
from tabnanny import verbose
from sklearn. datasets import load_boston        
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

     
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =90)

 #scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1))
model.summary()                     # () 주의 !!!!


# model.save("./_save/keras23_1_save_model.h5")     # 모델만 된다. 


model.save_weights("./_save/keras23_5_save_weights1.h5")


# import time

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, 
                              restore_best_weights=True)


hist = model.fit(x_train, y_train, 
                 epochs =100, batch_size = 32, 
                 verbose=1, 
                 validation_split = 0.2,
                callbacks=[earlyStopping])  

model.save_weights("./_save/keras23_5_save_weights2.h5")   

# model.save("./_save/keras23_3_save_model.h5")    

loss = model.evaluate(x_test, y_test)         
print('loss : ', loss)

# model = load_model("./_save/keras23_3_save_model.h5")   # 가중치와 모델이 저장이 된다. 

4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
# print('걸린시간 : ', end_time)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어:', r2)

# loss :  17.572927474975586
# 걸린시간 :  6.888061046600342
# r2스코어: 0.8013031488260934






