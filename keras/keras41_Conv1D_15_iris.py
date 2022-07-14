

from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
print(datasets.DESCR)
print(datasets.feature_names)
print(x)
print(y)
print(x.shape,y.shape) # (150, 4) (150,)
print("y의 라벨값 : ", np.unique(y))  # y의 라벨값 :  [0 1 2]
y = to_categorical(y) # https://wikidocs.net/22647 케라스 원핫인코딩
# print(y)
# print(y.shape) #(150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

print(x_train.shape,x_test.shape)  #(120, 4) (30, 4)

x_train = x_train.reshape(120, 4,1)
x_test = x_test.reshape(30, 4,1)

#2. 모델

model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(Conv1D(128, 2, input_shape=(4,1)))
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
model.add(Dense(3, activation='softmax'))


import time

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
               metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                               restore_best_weights=True)   

start_time = time.time()

model.fit(x_train, y_train, epochs=200, batch_size=32,             
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)

end_time = time.time() - start_time

# model.save("./_save/keras23_11_load_iris.h5")
# model = load_model("./_save/keras23_11_load_iris.h5")

#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
print("걸린시간 : ", end_time)
results= model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
# # r2 = r2_score(y_test,y_predict)                         #회귀모델 / 분류모델에서는 r2를 사용하지 않음 
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)
# # print(y_predict)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("걸린시간 : ", end_time)
print('r2 스코어 :', r2)

# y_predict = model.predict(x_test)


# y_predict = np.argmax(y_predict, axis= 1)

# y_predict = to_categorical(y_predict)


# acc= accuracy_score(y_test, y_predict) 
# print('acc스코어 : ', acc) 

# 전
# loss :  0.7270562648773193
# accuracy :  0.3333333432674408
# 걸린시간 :  5.156307220458984
# r2 스코어 : -0.4786705150196812

# 후
# loss :  0.7270562648773193
# accuracy :  0.3333333432674408
# 걸린시간 :  0.0
# r2 스코어 : -0.4786705150196812

# loss :  0.0605427548289299
# accuracy :  1.0
# 걸린시간 :  15.787106275558472
# r2 스코어 : 0.9636480821065501

# Conv1D 
# loss :  0.6240972876548767
# accuracy :  0.6666666865348816
# 걸린시간 :  10.761095762252808
# r2 스코어 : 0.4183097998794607