from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,6,7,9,10,13,14,15,17,8,16,23,24,26,29,27,30])


from sklearn.model_selection import train_test_split     
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size =0.7,                                
    shuffle=True, 
    random_state =66)
 




#2. 모델구성
model = Sequential()
model.add(Dense(15, input_dim=1))
model.add(Dense(80))
model.add(Dense(85))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(11))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(41))
model.add(Dense(41))
model.add(Dense(31))
model.add(Dense(31))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(15))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss ='mae', optimizer='adam')
model.fit(x_train, y_train, epochs =100, batch_size = 1)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)

print('r2 스코어 :', r2)


# loss :  1.758468508720398
# r2 스코어 : 0.9130251249876765
#######################################
# loss :  0.7509753108024597
# r2 스코어 : 0.9185493263854302
########################################


# loss :  8.309121131896973
# r2 스코어 : 0.8150289967048879

# loss :  1.8149334192276
# r2 스코어 : 0.8055068161149022

# loss :  2.4177300930023193
# r2 스코어 : 0.7652314244360012

# import matplotlib.pyplot as plt

#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지마
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상 
#8. loss지표는 mse, mae
# [실습시작]

# r2_score = 0과 1 사이의 값을 가지며 1에 가까울수록 선형회귀 
# 모델이 데이터에 대하여 높은 연관성을 가지고 있다고 해석. 