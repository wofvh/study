from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from sqlalchemy import true
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import time
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 

#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target

print (x.shape, y.shape)                        # (1797 ,64)
print ( np.unique(y,return_counts=True))        # [0,1,2,3,4,5,6,7,8,9]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=58525
                                                    )
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

print(x_train.shape,x_test.shape)  #(1437, 64) (360, 64)

x_train = x_train.reshape(1437, 64,1)
x_test = x_test.reshape(360, 64,1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(LSTM(350, input_shape=(64,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(10, activation='softmax'))            # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
                                                                 # ex 70, 20, 10 -> 0.7, 0.2, 0.1
                                                                 
                                                                 
# input1 = Input(shape=(64,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(80, activation='relu')(dense2)
# dense4 = Dense(15, activation='relu')(dense3)
# output1 = Dense(10, activation='softmax')(dense4)
# model = Model(inputs = input1, outputs = output1)

# model.summary()                       
# Total params: 11,205          
            
                                          
start_time = time.time()
#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/7digit/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'7digit_',date, '_', filename])
                      ) 


earlyStopping = EarlyStopping(monitor='val_loss', patience=80, mode='auto', verbose=1, 
                              restore_best_weights=True)   

model.fit(x_train, y_train, epochs=500, batch_size=3200,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)


end_time = time.time() - start_time

#  model.save("./_save/keras23_12_load_wine.h5")
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
print("걸린시간 :",end_time)

#전 
# acc :  0.9166666666666666
# 걸린시간 : 16.56109356880188
# loss :  0.27097490429878235

#후
# acc :  0.9166666666666666
# 걸린시간 : 0.0
# loss :  0.27097490429878235

#LSTM
# acc :  0.7583333333333333
# 걸린시간 : 239.05154967308044
# loss :  0.9018072485923767