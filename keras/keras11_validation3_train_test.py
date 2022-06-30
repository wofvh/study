from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
       
from sklearn.model_selection import train_test_split

#1. 데이터 
x = np.array(range(1,17))
y = np.array(range(1,17))       


x_train, x_rem, y_train, y_rem = train_test_split(
    x,y, train_size =0.65,                                
    shuffle=True, 
    random_state =66)

x_val, x_test, y_val, y_test =  train_test_split(
    x_rem, y_rem, train_size= 0.5,
    shuffle=True, 
    random_state =66)

print(x_train)
print(x_test)
print(x_val)

# 10 train 
# 3  test
# 3  validation


# #2. 모델
# model = Sequential()
# model.add(Dense(5, input_dim =1))
# model.add(Dense(3))
# model.add(Dense(1))

# #3. 컴파일, 훈련

# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train, y_train, epochs=100, batch_size=1,
#           validation_data=(x_val, y_val))                   # val_loss 는 일반loss값보다 높게 나온다.(좋지않다.)


# #4. 평가, 예측 
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

# result = model.predict([17])
# print("17의 예측값", result)

