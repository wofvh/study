import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터 
# import numpy as np
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])  # RNN = 3차원. 
y = np.array([4,5,6,7,8,9,10])                                  # (n,3,1) 1을 붙여준다.세번째 값은 자르는 단위.
                                                                # input_shape =(행, 열 , 자르는단위)
print(x.shape,y.shape)

x =x.reshape(7,3,1)
print(x.shape)

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3,                                
#     shuffle=True, random_state =58525)
# import numpy as np
# dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# def split_xy1(dataset, time_steps):                             # def 정의하겠다. 
#   x, y = list(), list()
#   for i in range(len(dataset)):
#     end_number = i + time_steps
#     if end_number > len(dataset) - 1:
#       break
#     tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
#     x.append(tmp_x)
#     y.append(tmp_y)
#   return np.array(x), np.array(y)

# x, y = split_xy1(dataset, 4)
# print(x, "\n", y)


#2. 모델
model = Sequential()
model.add(SimpleRNN(64, input_shape=(3,1)))    
# model.add(SimpleRNN(32))                                       # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(4, activation='swish'))
model.add(Dense(1, activation='swish'))                                           # erorr = ndim=3 3차원으로 바꿔라. 



#3. 컴파일,훈련
model.compile(loss='mse',optimizer='adam')

# earlystopping =EarlyStopping(monitor='loss', patience=40, mode='auto', 
#               verbose=1, restore_best_weights = True)     
        
# hist = model.fit(x_train, y_train, epochs=500,verbose=1,
#                  validation_split=0.2, callbacks=[earlystopping])

model.fit(x,y,epochs=1000)

#4. 평가,예측
loss = model.evaluate(x, y)
y_pred = np.array([8,9,10]).reshape(1, 3, 1)                 # 8,9,10을 reshape 하겠다.
result = model.predict(y_pred)
print('loss :', loss)
print('[8,9,10]의 결과', result)

# [8,9,10]의 결과 [[10.773127]]
# [8,9,10]의 결과 [[10.835447]]