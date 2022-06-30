from re import X
from tkinter import Y
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터 

x = np.array(range(1,17))
y = np.array(range(1,17))


x_train = x[:11]
y_train = y[:11]     
x_test = x[11:14]                          
y_test = y[11:14]
x_val = x[13:]                          
y_val = y[13:]


print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)
'''''
#2. 모델
model = Sequential()
model.add(Dense(5, input_dim =1))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))                   # val_loss 는 일반loss값보다 높게 나온다.(좋지않다.)


#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값", result)


'''