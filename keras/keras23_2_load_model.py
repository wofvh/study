
from tabnanny import verbose
from sklearn. datasets import load_boston        
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.callbacks import EarlyStopping
#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

     
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =90)

 #scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1))
model.summary()                     # () 주의 !!!!

model.save("./_save/keras23_1_save_model.h5")

model = load_model("./_save/keras23_1_save_model.h5")   # 저장한 모델 불러오기 

model.summary()


import time

#3 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, 
                              restore_best_weights=True)


model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=34, verbose=1, 
          validation_split=0.2, callbacks=[earlyStopping])


start_time = time.time()

hist = model.fit(x_train, y_train, epochs =100, batch_size = 32, verbose=1, validation_split = 0.2)     

end_time = time.time() - start_time

# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
print('걸린시간 : ', end_time)






