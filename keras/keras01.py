#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
#numpy는 고성능의 수치 계산을 위해 만들어진 라이브러리.
#효율적인 데이터 분석이 가능하도록 N차원의 배열 객체를 지원.
#array와 함께사용함.

#2. 모델구성
import tensorflow as tf      # import : 땡겨오다.
from tensorflow.keras.models import Sequential # sequential 순차적인
from tensorflow.keras.layers import Dense      # Dense      밀집도

# 

model = Sequential()
model.add(Dense(3, input_dim=1)) # input은 =1 첫번째 열 뒤에 포함
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')  # compile 엮다. #loss='mse'  optimizer='adam'(최적화)
model.fit(x, y, epochs=100)                 # model.fit > 훈련시키다.     epochs > 훈련횟수

#4. 평가, 예측
loss = model.evaluate(x, y)                 # 추출된값을 평가하는 과정
print('loss : ', loss)

result = model.predict([4])                 # 원하는 예측값을 얻기 위한 과정 
print("[4]의 예측값 : ", result)







