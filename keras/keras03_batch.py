import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# batch는 데이터양이 많을 때 array안에 값을 나누어 계산하는 방법.

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential    #tensorflow > keras > models 안에 Sequential
from tensorflow.keras.layers import Dense         #tensorflow > keras > layers 안에 Dense

model = Sequential()                              # add는 더한다는 의미로 input_dim=1 값은 첫 layers에만 포함한다.
model.add(Dense(3, input_dim=1))
model.add(Dense(50))
model.add(Dense(90))
model.add(Dense(7))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=3, batch_size=1)             #batch 사용법. batch_size가 2면 2개씩 3이면 3개씩 실행함.

#4. 평가, 예측
loss = model.evaluate(x, y)                         # 반드시는 필요없지만, 결과값을 보는 것 
print('loss : ', loss)                              # 최소의 loss > 최적의 wieght 결정됨.

result = model.predict([6])                         # 6을 예측
print("[6]의 예측값 : ", result)


# loss :  0.42466479539871216
# [6]의 예측값 :  [[6.0971785]]

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(50))
# model.add(Dense(90))
# model.add(Dense(7))
# model.add(Dense(1))


# #3. 컴파일, 훈련
# model.compile(loss='mse',optimizer='adam')
# model.fit(x, y, epochs=3, batch_size=1)