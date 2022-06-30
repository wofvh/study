
from sklearn. datasets import load_boston        
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target


x_train, x_rem, y_train, y_rem = train_test_split(
    x,y, train_size =0.65,                                
    shuffle=True, 
    random_state =66)

x_val, x_test, y_val, y_test =  train_test_split(
    x_rem, y_rem, train_size= 0.5,
    shuffle=True, 
    random_state =66)
 
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss ='mae', optimizer='adam')
model.fit(x_train, y_train, epochs =300, batch_size = 2,  validation_data=(x_val, y_val))

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)

print('r2 스코어 :', r2)


# loss :  3.543933153152466
# r2 스코어 : 0.6814641571250657

# model = Sequential()
# model.add(Dense(15, input_dim=13))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(60))
# model.add(Dense(20))
# model.add(Dense(1))

# loss :  3.122990608215332
# r2 스코어 : 0.7129551586526496

# #2. 모델구성
# model = Sequential()
# model.add(Dense(15, input_dim=13))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(60))
# model.add(Dense(20))
# model.add(Dense(1))

# #3 컴파일, 훈련
# model.compile(loss ='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs =300, batch_size = 2)




# #3 컴파일, 훈련
# model.compile(loss ='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs =200, batch_size = 2)





# print(x)
# print(y)
# print(x.shape) # (506,13)
# print(y.shape) # (506,)

# 평가 > x_test, y_test 

# 예측 > y_test


# print(datasets.feature_names)
# # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
# print(datasets.DESCR)

