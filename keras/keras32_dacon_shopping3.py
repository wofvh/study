
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization

# ###########################폴더 생성시 현재 파일명으로 자동생성###########################################
# import inspect, os
# a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
# print(a)
# print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
# print(a.split("\\")[-1]) #현재 파일 명
# current_name = a.split("\\")[-1]
# ##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################


#1. 데이터
path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv',     # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)            # index_col=n n번째 컬럼을 인덱스로 인식
Weekly_Sales = train_set[['Weekly_Sales']]
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv',        # 예측                
                       index_col=0)
print(test_set)
print(test_set.shape)                           # (180, 11)

print(train_set.columns)
print(train_set.info())                         # info 정보출력
print(train_set.describe())                     # describe 평균치, 중간값, 최소값 등등 출력

train_set.isnull().sum().sort_values(ascending=False)           #sort_values(ascending=False) 내림차순
test_set.isnull().sum().sort_values(ascending=False)            #sort_values(ascending=True) 오름차순

######## 년, 월 ,일 분리 ############

train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.Date)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.Date)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.Date)]

test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.Date)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.Date)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.Date)]

train_set.drop(['Date','Weekly_Sales'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
test_set.drop(['Date'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)
print(test_set)

#원핫인코더#

df = pd.concat([train_set, test_set])
print(df)

alldata = pd.get_dummies(df, columns=['day','Store','month', 'year', 'IsHoliday'])
print(alldata)

train_set2 = alldata[:len(train_set)]
test_set2 = alldata[len(train_set):]

print(train_set2)
print(test_set2)

# 결측치 처리#

train_set2 = train_set2.fillna(0)
test_set2 = test_set2.fillna(0)

print(train_set2)
print(test_set2)

train_set2 = pd.concat([train_set2, Weekly_Sales],axis=1)               # concat 열이 다를때 합치는 방법 
print(train_set2)

x = train_set2.drop(['Weekly_Sales'], axis=1)
y = train_set2['Weekly_Sales']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


test_set2 = scaler.transform(test_set2)
print(test_set2.shape)


print(test_set2)

# 2. 모델구성
model = Sequential()
model.add(Dense(6400, activation='relu', input_dim=77))
model.add(Dropout(0.3))
model.add(Dense(3200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1600, activation='LeakyReLU'))
model.add(Dropout(0.3))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='LeakyReLU'))
model.add(Dropout(0.3))
model.add(Dense(15, activation='LeakyReLU'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=3000, batch_size=256,
                 validation_split=0.3,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

print("======================출력=============================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)


print("======================제출=============================")
# print(test_set2)

y_summit = model.predict(test_set2)
# print(y_summit)
# print(y_summit.shape) # (180, 1)

submission_set = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(submission_set)

submission_set['Weekly_Sales'] = y_summit
# print(submission_set)

submission_set.to_csv(path + 'submission4.csv', index = True)

