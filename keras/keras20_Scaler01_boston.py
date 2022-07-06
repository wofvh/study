from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 대문자 class  암시가능.
from sklearn.preprocessing import MaxAbsScaler, RobustScaler    # > 과제 

import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



datasets = load_boston()
x = datasets.data
y = datasets.target

print(y)

# print(np.min(x))
# print(np.max(x))
# x = (x - np.min(x) / (np.max(x)) - np.min(x))  # 0~1 사이가 된다. 
# print(x[:10])

# scaler는 train_test_split 후에
# 전체데이터를 하는 것이 아니다. train 데이터만 scaler 하고 범위 밖에 것을 평가한다.  
# test를 train과 동일한 규칙으로 변환시킨다. 
# train 데이터만 따로 잡아서 돌리고, 그 값의 수식데로 test를 다시 수식을 한다. 1이상나온 것을 평가 한다. 
# test / val >> fit(), transform / test는 전체데이터로 돌리는 것이 아니다. 
# 전처리는 열별로 해야한다.    

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
    )

scaler = MinMaxScaler()
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
model.add(Dense(10, activation= 'linear', input_dim=13))
model.add(Dense(80, activation= 'linear'))
model.add(Dense(90, activation= 'linear'))
model.add(Dense(25, activation= 'relu'))        # relu 강력한놈
model.add(Dense(85, activation= 'linear'))
model.add(Dense(25, activation= 'linear'))      # linear = 기본값으로 생략 가능(회귀모델) 
model.add(Dense(1, activation= 'linear'))      # sigmoid = 0~1 사이로 숫자를 축소해줌. 아웃풋에 sigmoid 입력.
                                                # 회귀모델은 output = linear 자연수치 그데로 나와야 함. 디폴트.
                                                # 분류모델은 이진 > sigmoid / 


import time

#3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam',
              metrics=['mae'])                           # 이진분류 binary_crossentropy 반올림.
                                                                # 회귀 - mse,mae ~ / 이진 binary_crossentropy
                                                                # 분류모델 loss에 accuracy(정확도) 같이씀.
                                                                # 2개 이상은 list           
                                                                # 'mse'는 분류모델에서는 잘 맞지 않는다. 
                                                                # 회귀모델 > mitrics=['mae']
                                                                
from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping =EarlyStopping(monitor='val_loss', patience=50, mode='auto', 
              verbose=1, restore_best_weights = True)          
            
start_time = time.time()

hist = model.fit(x_train, y_train, epochs =500, batch_size = 30, 
                 verbose=1, 
                 validation_split = 0.2,
                 callbacks = [earlystopping])      # callbacks으로 불러온다 erlystopping   

end_time = time.time() - start_time

#4 평가 예측
print(x_test.shape)
print(y_test.shape)

loss = model.evaluate(x_test, y_test)




print("loss : ", loss)

print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(x_test, y_predict)

print('r2 스코어 :', r2)



#1. scaler 하기전 
# loss :  0.07886183261871338
# 걸린시간 :  32.08338141441345
# r2 스코어 : 0.6699231715644725

#2. minmaxscaler


#3. standardscaler 

#4. MaxAbsScaler


#5. RobustScaler
