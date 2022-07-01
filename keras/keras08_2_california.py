# R2 055~0.6 이상

from pyexpat import model
from tabnanny import verbose
from sklearn. datasets import fetch_california_housing     
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape)           #(20640, 8)  # 값 찾기. 
print(y.shape)           #(20640, ) 
                              # - MedInc        median income in block group
                              # - HouseAge      median house age in block group
                              # - AveRooms      average number of rooms per household
                              # - AveBedrms     average number of bedrooms per household
                              # - Population    block group population
                              # - AveOccup      average number of household members
                              # - Latitude      block group latitude
                              # - Longitude     block group longitude

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size =0.9,                                
    shuffle=True, 
    random_state =525)
 
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss ='mae', optimizer='adam')
model.fit(x_train, y_train, epochs =30, batch_size = 2, verbose=1)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('r2 스코어 :', r2)



# loss :  0.551124632358551
# r2 스코어 : 0.5667785096003797

# x_train, x_test, y_train, y_test = train_test_split(
#     x,y, train_size =0.9,                                
#     shuffle=True, 
#     random_state =525)
 
# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(60))
# model.add(Dense(20))
# model.add(Dense(1))

# #3 컴파일, 훈련
# model.compile(loss ='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs =30, batch_size = 2)