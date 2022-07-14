import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN ,LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y= np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape,y.shape)   # (13, 3) (13,)

#2. 모델구성
model = Sequential()
model.add(LSTM(10 ,input_shape=(3,1)))       # (N,3,1) > (N,3,10)
# model.add(LSTM(5,return_sequences=False))
model.add(Dense(1))
model.summary()                     
# lstm 2개 엮은거 테스트 


x = x.reshape(13,3,1)

# x_predict =np.array([50,60,70])
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(GRU(350, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='LeakyReLU'))
model.add(Dense(32, activation='LeakyReLU'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(4, activation='LeakyReLU'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()

#3. 컴파일,훈련
model.compile(loss='mse',optimizer='adam')

# earlystopping =EarlyStopping(monitor='loss', patience=40, mode='auto', 
#               verbose=1, restore_best_weights = True)     
        
# hist = model.fit(x_train, y_train, epochs=500,verbose=1,
#                  validation_split=0.2, callbacks=[earlystopping])

model.fit(x,y,epochs=280)

#4. 평가,예측
loss = model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1)                 # 8,9,10을 reshape 하겠다.
result = model.predict(y_pred)
print('loss :', loss)
print('[50,60,70]의 결과', result)


# [8,9,10]의 결과 [[80.155266]]

# loss : 0.0016995823243632913
# [8,9,10]의 결과 [[80.05531]]

# loss : 0.001789769041351974
# [8,9,10]의 결과 [[80.51578]]
