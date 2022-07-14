from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target #데이터를 리스트 형태로 불러올 때 함

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)

from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(datasets.feature_names)
# print(datasets.DESCR)
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)

x_train = x_train.reshape(404, 13,1)
x_test = x_test.reshape(102, 13,1)

print(x_train.shape, x_test.shape)

#2. 모델구성
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(Conv1D(128, 2, input_shape=(13,1)))
model.add(Flatten())
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()


#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='mae', optimizer='adam')

import time

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=50, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=2)

end_time = time.time() - start_time

# #4. 평가,예측
print("=========================1.기본출력========================")
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)
print('시간 :',end_time)

#DNN
# loss : 9.402571678161621
# r2 : -0.5285233444774788

# dropout
# loss : 9.012863159179688
# r2 : -0.40762565852655896

# LSTM
# r2 : -0.006260202196402664

# Conv1D
# loss : 2.5506229400634766
# r2 : 0.7800906920134872
# 시간 : 13.291000127792358