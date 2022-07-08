from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout



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



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation='linear'))
model.add(Dropout(0.4))                                  #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dense(100, activation='relu'))     
model.add(Dropout(0.3))                                  
model.add(Dense(80, activation='relu'))    
model.add(Dropout(0.2))                                  
model.add(Dense(15, activation='relu'))               
model.add(Dense(3, activation='softmax'))             # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
#                                                                  # ex 70, 20, 10 -> 0.7, 0.2, 0.1
# input1 = Input(shape=(13,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(80, activation='relu')(dense2)
# dense4 = Dense(15, activation='relu')(dense3)
# output1 = Dense(3, activation='softmax')(dense4)
# model = Model(inputs = input1, outputs = output1)

import time
start_time = time.time()

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/6wine/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'6wine_',date, '_', filename])
                      ) 


earlyStopping = EarlyStopping(monitor='val_loss', patience=80, mode='auto', verbose=1, 
                              restore_best_weights=True)   

model.fit(x_train, y_train, epochs=500, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


# model.save("./_save/keras23_12_load_wine.h5")
# model = load_model("./_save/keras23_12_load_wine.h5")


#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

results= model.evaluate(x_test, y_test)

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
