# multi layer 
# 열 = 특성 = 피처 = 컬럼 많이 있을수록 좋은 데이터가 나온다.

# <loss와 W 관계> loss 작으수록 좋다
# loss가 가장 낮은 지점이 w이 가장 좋은지점. 

# 데이터 - 모델구성 - 컴파일구성 - 평가예측

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])   # -> y= w1x1 + w2x2 +b
y = np.array([11,12,13,14,15,16,17,18,19,20])    # (10,)

print (x.shape)  #(2,10)
print (y.shape)  #(10,)

# x = x.transpose()                     # transpose 행과열의 위치변경 방법1 
x = x.T                                 # transpose 행과열의 위치변경 방법2
print(x)
print(x.shape)   #(10,2)

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))        # 열,컬럼,피처,특성 값이 들어감 
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(1))                     # 마지막 레이어값에 y값(벡터)이 들어가야함 

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1) 

# 4. 평가, 예측
loss = model.evaluate(x, y)                        
print('loss : ', loss) 

result = model.predict([[10,1.4]])              # (2,) 오류  > []를 추가해서 열을 2개를 만든다. 
print("[10, 1.4]의 예측값", result)



 # [1,2,3] 1벡터, 3스칼라 
 
 # [[1,2,3],[4,5,6]] 1행렬/1메트릭스
 
 # [[1,2,3],[4,5,6]]
 # [[4,5,6],[3,2,1]] 2행렬 > Tensor
  
 # 행렬 = 메트릭스 
 # 개체 = 스칼라 / 묶음 = 벡터  
 
 # [1,2,3] 3스칼라 1벡터
 
 # [[1,2,3]] 1행3열  
 
 # input_dim=2 행의 갯수가 2개 ex) #[[1,2,3],[4,5,6]] < 열-피처-컬럼의 개수가 dim의 값.
 # [1,2,3] = (3,) < 1벡터 = dim=1
 # [[1,2,],[3,4],[5,6]] 3행 2열 (3,2) < 가장적은열의 개수
 # [[[1,2,3,4,5]]] (1,1,5)          < 참고[]가 남으면 앞에 1을 표시 
 # 행무시 열우선 !

# loss :  0.03149569779634476
# [10, 1.4]의 예측값 [[20.222898]]

# model = Sequential()
# model.add(Dense(5, input_dim=2))        
# model.add(Dense(80))
# model.add(Dense(100))
# model.add(Dense(25))
# model.add(Dense(1))                    

# # 3. 컴파일 훈련
# model.compile(loss='mse',optimizer='adam')
# model.fit(x, y, epochs=200, batch_size=1)