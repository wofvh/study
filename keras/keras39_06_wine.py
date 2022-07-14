from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, LSTM



import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sqlalchemy import true
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

print (x.shape, y.shape)                                  # (178 ,13)
print (np.unique(y,return_counts=True))                   #  0,1,2

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=58525
                                                    )
scaler = RobustScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

print(x_train.shape,x_test.shape)  #(142, 13) (36, 13)

x_train = x_train.reshape(142, 13,1)
x_test = x_test.reshape(36, 13,1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(LSTM(350, input_shape=(13,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(3, activation='softmax'))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()  

import time
start_time = time.time()

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=80, mode='auto', verbose=1, 
                              restore_best_weights=True)   

model.fit(x_train, y_train, epochs=500, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


# model.save("./_save/keras23_12_load_wine.h5")
# model = load_model("./_save/keras23_12_load_wine.h5")


#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

results= model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)

print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
print(y_predict)
y_predict = to_categorical(y_predict)


acc= accuracy_score(y_test, y_predict) 
print('acc : ', acc) 
print("걸린시간 : ", end_time)

#1. 하기전 
# acc :  0.9444444444444444
# 걸린시간 :  10.174596071243286
# loss :  0.12215370684862137

#2. 후
# acc :  0.9444444444444444
# 걸린시간 :  0.0
# loss :  0.12215370684862137


# LSTM
# acc :  0.8888888888888888
# 걸린시간 :  105.68571662902832