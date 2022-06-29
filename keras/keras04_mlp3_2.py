import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10), range(21, 31),range(201, 211)])

print(x.shape)  #(3,10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)  #(3,10)

x = x.T                                 # transpose 행과열의 위치변경 방법2
print(x)
print(x.shape) 

y = y.T                                 # transpose 행과열의 위치변경 방법2
print(y)
print(y.shape)                          # x와 y의 행렬배열을 맞춰야 함.

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))        # 열,컬럼,피처,특성 값이 들어감 *열이 3개일 때 input_dim=3 으로 계산.
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(3))                     # y가 2개일때, Dense(2)으로 변환.

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1) 

# 4. 평가, 예측
loss = model.evaluate(x, y)                        
print('loss : ', loss) 

result = model.predict([[9, 30, 210]])              
print("[9, 30, 210]의 예측값", result)  

# loss :  0.00028317421674728394
# [9, 30, 210]의 예측값 [[9.992      1.886061   0.01533388]]

# loss :  1.3751102301284845e-07
# [9, 30, 210]의 예측값 [[9.9995909e+00 1.8997842e+00 9.8300632e-05]]