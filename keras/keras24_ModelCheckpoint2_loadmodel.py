
from gc import callbacks
from tabnanny import verbose
from sklearn. datasets import load_boston 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

#1. 데이터
datasets =  load_boston()
x, y  = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))


#2. 모델구성

# input1 = Input(shape=(13,))                          # 컬럼3개를 받아드린다.
# dense1 = Dense(100)(input1)                            # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(50, activation='relu')(dense1)
# dense3 = Dense(30, activation='sigmoid')(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs = input1, outputs = output1)

# # #3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.
earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath='./_save/_ModelCheckPoint/keras24_ModelCheckPoint.hdf5'
                      ) 
import time                               
start_time = time.time()
        
# hist = model.fit(x_train, y_train, epochs =100, batch_size = 32, 
#                  verbose=1, validation_split = 0.2,
#                  callbacks = [earlystopping, mcp])                                    # callbacks으로 불러온다 erlystopping   

end_time = time.time() - start_time

model = load_model('./_save/_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('r2 스코어 :', r2)
print('걸린시간 :', end_time)


# loss :  105.92610931396484
# r2 스코어 : -0.28213294996153215
# 걸린시간 : 9.003702640533447

# loss :  105.92610931396484
# r2 스코어 : -0.28213294996153215
# 걸린시간 : 0.0

