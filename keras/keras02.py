import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,5,4])                   #데이터변경
y = np.array([1,2,3,4,5])

#numpy는 고성능의 수치 계산을 위해 만들어진 라이브러리.
#효율적인 데이터 분석이 가능하도록 N차원의 배열 객체를 지원.
#array와 함께사용함.

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=1))           # dence(?) ?= road
model.add(Dense(45))                        # 줄은 layers
model.add(Dense(120))
model.add(Dense(75))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])                 # 예측값 변경하고 예측값과 비슷한 결과만들기. 
print("[6]의 예측값 : ", result)

# loss :  0.3989199697971344
# [6]의 예측값 :  [[6.0181894]]

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(50))
# model.add(Dense(90))
# model.add(Dense(50))
# model.add(Dense(7))
# model.add(Dense(1))

# model.fit(x, y, epochs=50)

# loss :  0.3982747197151184
# [6]의 예측값 :  [[6.0116262]]

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(50))
# model.add(Dense(90))
# model.add(Dense(50))
# model.add(Dense(7))
# model.add(Dense(1))

# model.fit(x, y, epochs=45)

# loss :  0.4005369246006012
# [6]의 예측값 :  [[6.0194273]]

# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(80))
# model.add(Dense(90))
# model.add(Dense(50))
# model.add(Dense(7))
# model.add(Dense(1))

# model.fit(x, y, epochs=50)