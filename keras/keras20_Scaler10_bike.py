# 캐글 자전거 문제풀이
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential           #
from tensorflow.python.keras.layers import Dense
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

'''                        
print(train_set)
print(train_set.shape) # (10886, 12)
                  
print(test_set)
print(test_set.shape) # (6493, 9)
print(test_set.info()) # (715, 9)
print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력
'''


######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)
print(test_set)

##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)
y = train_set['count'] 
print(y)
print(y.shape) # (10886,)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )

#scaler = MaxAbsScaler()
scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=12))
model.add(Dense(100, activation='relu'))
model.add(Dense(80))
model.add(Dense(50, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1))

import time
start_time = time.time()
#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=500, batch_size=150, verbose=1,validation_split=0.2, callbacks=[earlyStopping])

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
y_predict = model.predict(x_test)
def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_test, y_predict)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)
# loss :  20049.21484375
# RMSE :  140.3344816795905
# r2스코어 :  0.3978910778053413
##################activation전후#################
# loss :  1027.9700927734375
# RMSE :  42.40323210832085
# r2스코어 :  0.9450276636367779
y_summit = model.predict(test_set)
print(y_summit)
print(y_summit.shape) # (6493, 1)
submission_set = pd.read_csv(path + 'Submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(submission_set)
submission_set['count'] = y_summit
print(submission_set)
submission_set.to_csv(path + 'submission.csv', index = True)

#1. scaler 하기전 
# loss: 2150.8689
# RMSE :  51.364839254484586
# r2스코어 :  0.9193362998950112
# 걸린시간 :  104.75211715698242

#2. minmaxscaler
# loss: 6822311.5000
# RMSE :  45.27189771343862
# r2스코어 :  0.9373380836853957
# 걸린시간 :  104.68631792068481

#3. standardscaler 
#  loss: 54981.5273
# RMSE :  47.56385690860632
# r2스코어 :  0.9308327672555263
# 걸린시간 :  105.26469230651855

#4. MaxAbsScaler
# loss: 6752392.0000
# RMSE :  45.53995911490976
# r2스코어 :  0.9365938263692993
# 걸린시간 :  99.33333206176758

#5. RobustScaler
# loss: 73175.1484
# r2스코어 :  0.9421554411680493
# 걸린시간 :  100.76702523231506

