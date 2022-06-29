import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라 

from sklearn.model_selection import train_test_split
x_tratin, x_test, y_train, y_text = train_test_split(
x,y, test_size=0.3, 
train_size=0.7,
random_state=99)

# x_train = x[1,3,5,7,8,9,10]   
# x_test = x[]    
# y_train = y[:7]    
# y_test = y[7:] 

print(x_tratin)
print(x_test)
print(y_train)
print(y_text)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x_tratin, y_train, epochs =100, batch_size = 1)

#4 평가 예측
loss = model.evaluate(x_test, y_text)
print("loss : ", loss)

result = model.predict([11])
print("[11의 예측 값 : ", result)

