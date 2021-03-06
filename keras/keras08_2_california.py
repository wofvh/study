from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=66)


# print(x)
# print(y)
# print(x.shape, y.shape)   #(20640, 8) (20640,)
# print(datasets.feature_names)
# print(datasets.DESCR)

#모델
model = Sequential()
model.add(Dense(5, input_dim=8))
model.add(Dense(68))
model.add(Dense(115))
model.add(Dense(98))
model.add(Dense(118))
model.add(Dense(121))
model.add(Dense(98))
model.add(Dense(1))

#컴파일 훈련model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=489, batch_size=10)
#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

#결과
# loss :  0.6192092895507812
# r2스코어 : 0.5343580504592822