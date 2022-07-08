from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
#1. 데이터

datasets = load_breast_cancer()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
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


#2. 모델구성
model = Sequential()
model.add(Dense(128, activation= 'linear', input_dim=30))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))        # relu 강력한놈
model.add(Dense(8, activation= 'relu'))
model.add(Dense(4, activation= 'relu'))      # linear = 기본값 / 생략 가능(회귀모델) 
model.add(Dense(1, activation= 'sigmoid'))      # sigmoid = 0~1 사이로 숫자를 축소해줌. 아웃풋에 sigmoid 입력.
                                                # 회귀모델은 output = linear 자연수치 그데로 나와야 함. 디폴트.
                                                # * 분류모델은 이진 > 마지막 activation = sigmoid 

# input1 = Input(shape=(30,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(50, activation='relu')(dense1)
# dense3 = Dense(30, activation='sigmoid')(dense2)
# output1 = Dense(1)(dense3)

# model = Model(inputs = input1, outputs = output1)

import time

#3 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])                      # * 이진분류 할 때 binary_crossentropy 반올림.
                                                                # 회귀 - mse,mae ~ / 이진 binary_crossentropy
                                                                # 분류모델 loss에 accuracy(정확도) 같이씀.
                                                                # 2개 이상은 list           
                                                                # 'mse'는 분류모델에서는 잘 맞지 않는다. 
                                                                # 회귀모델 > mitrics=['mae']
                                                                # 분류모델 > metrics=['accuracy','mse'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/4cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'4cancer_',date, '_', filename]))                                                                
                                                                
# from tensorflow.python.keras.callbacks import EarlyStopping
# earlystopping =EarlyStopping(monitor='loss', patience=50, mode='min', 
#               verbose=1, restore_best_weights = True)          
            

start_time = time.time()

hist = model.fit(x_train, y_train, epochs =500, batch_size = 32, 
                 verbose=1, 
                 validation_split = 0.2,
                 callbacks = [earlystopping, mcp])      # callbacks으로 불러온다 erlystopping   

end_time = time.time() - start_time

# model.save("./_save/keras23_10_load_cancer.h5")
# model = load_model("./_save/keras23_10_load_cancer.h5")

#4 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

y_predict[(y_predict<0.5)] = 0                        # 반올림하기 y_predict가 0.5 보다 높으면 1 아니면 0 
y_predict[(y_predict>0.5)] = 1
print(y_predict, y_test)

from sklearn.metrics import accuracy_score 
acc = accuracy_score(y_test, y_predict)



print('acc 스코어 :', acc)
# # print(y_predict)


print("걸린시간 : ", end_time)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict)

# print('r2 스코어 :', r2)


#1. 전 
# loss :  [9.922481536865234, 0.35672515630722046, 1.4179694652557373]        
# 걸린시간 :  4.486129283905029
# r2 스코어 : -5.179261591754954

#2. 후
# loss :  [9.922481536865234, 0.35672515630722046, 1.4179694652557373]        
# 걸린시간 :  0.0
# r2 스코어 : -5.179261591754954




