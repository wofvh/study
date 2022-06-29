
# 예측[[9]] > 예상 [[10,1.9,0]]

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10)])

print(x.shape)  #(10,)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)  #(3,10)

x = x.T                               
print(x)
print(x.shape) 

y = y.T                                 
print(y)
print(y.shape)                          

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))        
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(3))                   

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam') 
model.fit(x, y, epochs=200, batch_size=1) 

# 4. 평가, 예측
loss = model.evaluate(x, y)                        
print('loss : ', loss) 

result = model.predict([[9]])              
print("[9]의 예측값", result)  

# loss :  3.639236806872448e-13
# [9]의 예측값 [[ 1.0000000e+01  1.8999996e+00 -7.5995922e-07]]