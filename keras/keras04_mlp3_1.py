
# [실습] 만들기. 
# 예측 9,30,210 > 예상 값 10,1.9

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10), range(21, 31),range(201, 211)])
print(x)
exit()
print(x.shape)  #(3,10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])
print(y.shape)  #(2,10)

x = x.T                                 # transpose / x.T 행과열의 위치변경 방법2
print(x)
print(x.shape) 

y = y.T                                 # transpose / x.T 행과열의 위치변경 방법2
print(y)
print(y.shape)                          # x와 y의 행렬배열을 맞춰야 함.

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))        # 열,컬럼,피처,특성 값이 들어감 *열이 3개일 때 input_dim=3 으로 계산.
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(2))                     # y가 2개일때, Dense(2)으로 변환.

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1) 

# 4. 평가, 예측
loss = model.evaluate(x, y)                        
print('loss : ', loss) 

result = model.predict([[9, 30, 210]])              
print("[9, 30, 210]의 예측값", result)            

# loss :  2.587194671832549e-07
# [9, 30, 210]의 예측값 [[9.999327  1.9004134]]

# range(stop)
# range(10)은 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 숫자를 생성한다.
# 마지막 숫자 10(stop)은 포함되지 않는다.
# (range 함수의 결과를 바로 확인하기 위해 리스트(list)로 변환)

# range(start, stop)
# range(1, 11)은 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 숫자를 생성
# 인자를 2개 전달하는 경우 첫번째 인자는 시작하는 숫자가 된다.

# range(start, stop, step)
# range(0, 20, 2)
# 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
# 마지막 인자 step은 숫자의 간격을 나타낸다.
# range(20, 0, -2)
# 20, 18, 16, 14, 12, 10, 8, 6, 4, 2

# step으로 음수를 지정할 수 있다.