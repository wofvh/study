from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.datasets import load_diabetes
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
    )

scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)

print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)




#2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=10))
# model.add(Dense(85))
# model.add(Dense(100))
# model.add(Dense(80,activation='relu'))
# model.add(Dense(15))
# model.add(Dense(1))

# input1 = Input(shape=(10,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(50, activation='relu')(dense1)
# dense3 = Dense(30, activation='sigmoid')(dense2)
# output1 = Dense(1)(dense3)

# model = Model(inputs = input1, outputs = output1)

import time
start_time = time.time()
model = load_model("./_save/keras23_9_load_diabet.h5")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1, 
                              restore_best_weights=True)

# hist = model.fit(x_train, y_train, epochs=10, batch_size=100,verbose=1,
#                  validation_split=0.2, callbacks=[earlyStopping])

end_time = time.time() - start_time

# model.save("./_save/keras23_9_load_diabet.h5")
# model = load_model("./_save/keras23_9_load_diabet.h5")

#4. 평가, 예측\
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print("걸린시간 : ", end_time)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

#1.  하기전 
# 걸린시간 :  1.2999696731567383
# loss :  [28918.599609375, 150.64321899414062]
# r2스코어 :  -3.6415536423300496

# 걸린시간 :  0.7596209049224854
# loss :  [28918.599609375, 150.64321899414062]
# r2스코어 :  -3.6415536423300496



