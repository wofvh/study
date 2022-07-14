import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM , GRU, Conv1D, Flatten


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping



#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


y_predict = np.array([50,60,70])
# print(x.shape,y.shape)   # (13, 3) (13,)

x = x.reshape(13,3,1)

#2. 모델
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(Conv1D(10, 2, input_shape=(3,1)))
model.add(Flatten())
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()

#3. 컴파일 
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=400)

#4. 평가,예측
loss = model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1)                 # 8,9,10을 reshape 하겠다.
result = model.predict(y_pred)
print('loss :', loss)
print('[50,60,70]의 결과', result)


# [50,60,70]의 결과 [[77.35976]]

# conv1D
# [50,60,70]의 결과 [[85.46665]]
