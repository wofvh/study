from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

# 캐글 자전거 문제풀이
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
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
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(y)
print(y.shape) # (1459,)
print(x_train.shape) #(8164, 12)
print(x_test.shape) #(2722, 12

x_train = x_train.reshape(8164, 6,2,1)
x_test = x_test.reshape(2722, 6,2,1)

# #scaler = MaxAbsScaler()
# scaler = RobustScaler()
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)
# x_test = scaler.transform(x_test)
# x_train = scaler.transform(x_train)
# print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
# print(np.max(x_train))      # 1
# print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
# print(np.max(x_test))
print(x_train.shape, x_test.shape)
# (8164, 6, 2, 1) (2722, 6, 2, 1)

# test_set = test_set.reshape(8164, 6,2,1)
print(test_set.shape,y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, kernel_size=(2,1),                                         # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(6,2,1), activation= 'relu'))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라                                                                        
model.add(Conv2D(64, (5,5), 
                 padding = 'same',        # 디폴트값(안준것과 같다.) 
                 activation= 'relu'))    # 출력(3,3,7)                                                     
model.add(Flatten()) # (N, 63)
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation= 'linear'))
model.summary()

# input1 = Input(shape=(12,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(80)(dense2)
# dense4 = Dense(50, activation='relu')(dense3)
# dense5 = Dense(15, activation='relu')(dense4)
# output1 = Dense(1, activation='relu')(dense5)

# model = Model(inputs = input1, outputs = output1)


import time
start_time = time.time()
#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 
filepath = './_ModelCheckPoint/10bike/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'10bike_',date, '_', filename])
                      ) 
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=150, verbose=1,
          validation_split=0.2, callbacks=[earlyStopping, mcp])

end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
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
# y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape) # (6493, 1)
# submission_set = pd.read_csv(path + 'Submission.csv', # + 명령어는 문자를 앞문자와 더해줌
#                              index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(submission_set)
# submission_set['count'] = y_summit
# print(submission_set)
# submission_set.to_csv(path + 'submission.csv', index = True)

# loss :  [1868.6226806640625, 29.775550842285156]
# r2스코어 :  0.7360222792243953
# 걸린시간 : 104.02831315994263

