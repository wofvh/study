

from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

y = to_categorical(y) # https://wikidocs.net/22647 케라스 원핫인코딩
# print(y)
# print(y.shape) #(150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)


#2. 모델

model = Sequential()
model.add(Dense(64, input_dim=4, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dense(32, activation='sigmoid')) 
model.add(Dropout(0.4))                                 # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
model.add(Dense(16, activation='relu'))   
model.add(Dropout(0.3))# relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
model.add(Dense(8, activation='relu'))  
model.add(Dropout(0.2)) 
model.add(Dense(4, activation='linear'))              
model.add(Dense(3, activation='softmax'))   

# input1 = Input(shape=(4,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(80, activation='relu')(dense2)
# dense4 = Dense(50, activation='relu')(dense3)
# dense5 = Dense(15, activation='relu')(dense4)
# dense6 = Dense(10, activation='relu')(dense5)
# output1 = Dense(3)(dense6)
# model = Model(inputs = input1, outputs = output1)


import time

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/5iris/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'5iris_',date, '_', filename])
                      ) 

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)   

start_time = time.time()

model.fit(x_train, y_train, epochs=200, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)

end_time = time.time() - start_time

# model.save("./_save/keras23_11_load_iris.h5")
# model = load_model("./_save/keras23_11_load_iris.h5")

#4. 평가, 예측
loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# results= model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])


y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = np.around(y_predict,0)

from sklearn.metrics import accuracy_score 
acc = accuracy_score(y_test, y_predict)

print('acc 스코어 :', acc)
print("걸린시간 : ", end_time)



