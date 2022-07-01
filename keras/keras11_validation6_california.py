
from sklearn. datasets import fetch_california_housing        
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data         # (20640, 8)
y = datasets.target       # (20640,)

print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size =0.625,                                
    shuffle=True, 
    random_state =58525)


 
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss ='mae', optimizer='adam')
model.fit(x_train, y_train, epochs =80, batch_size = 100,  validation_split=0.1)
#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)

print('r2 스코어 :', r2)

# loss :  0.6275011301040649
# r2 스코어 : 0.08922538243476485

# x_train, x_test, y_train, y_test = train_test_split(
#     x,y, train_size =0.625,                                
#     shuffle=True, 
#     random_state =58525)


 
# #2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=8))
# model.add(Dense(100))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(20))
# model.add(Dense(1))

# #3 컴파일, 훈련
# model.compile(loss ='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs =80, batch_size = 100,  validation_split=0.1)
#4 평가 예측
