import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]
             ,[9,8,7,6,5,4,3,2,1,0]])   # -> y= w1x1 + w2x2 +b
y = np.array([11,12,13,14,15,16,17,18,19,20])    # (10,)

print (x)  #(3,10)    # x = 행3, 열,10 
print (y.shape)  #(10,)

# x = x.transpose()                     # transpose 행과열의 위치변경 방법1 
x = x.T                                 # transpose 행과열의 위치변경 방법2
print(x)
print(x.shape)   #(10,3)
exit()

# [실습] 모델을 완성하시오.
# 예측값 [10,1.4,0]

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))        # 열,컬럼,피처,특성 값이 들어감 *열이 3개일 때 input_dim=3 으로 계산.
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

result = model.predict([[10,1.4,0]])              # (2,) 오류  > []를 추가해서 열을 2개를 만든다. (벡터의 열을 2개로 만들때만)
print("[10, 1.4, 0]의 예측값", result)             # 스칼라의 개수가 3개 = 열 3

# loss :  0.0005630630766972899
# [10, 1.4, 0]의 예측값 [[20.000399]]
# 2. 모델구성
# model = Sequential()
# model.add(Dense(5, input_dim=3))        # 열,컬럼,피처,특성 값이 들어감 *열이 3개일 때 input_dim=3 으로 계산.
# model.add(Dense(80))
# model.add(Dense(100))
# model.add(Dense(25))
# model.add(Dense(1))                     # 마지막 레이어값에 y값(벡터)이 들어가야함 

# # 3. 컴파일 훈련
# model.compile(loss='mse',optimizer='adam')
# model.fit(x, y, epochs=200, batch_size=1) 

# loss :  3.8903635868337005e-09
# [10, 1.4, 0]의 예측값 [[19.99999]]