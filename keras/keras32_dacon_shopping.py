from sqlite3 import Date
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sqlalchemy import true                                 #pandas : 엑셀땡겨올때 씀
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, LabelEncoder 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import Series, DataFrame
train = pd.read_csv('./_data/shopping/train.csv')
test = pd.read_csv('_data/shopping/test.csv')
sample_submission = pd.read_csv('_data/shopping/sample_submission.csv')

# train = pd.read_csv('./_data/shopping/train.csv')
# test =pd.read_csv('/_data/shopping/test.csv')

print(train)  # [6255 rows x 13 columns]
print(test)   # [180 rows x 12 columns]

#1. 데이터
path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv',               # + 명령어는 문자를 앞문자와 더해줌
                       index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                      index_col=0)

data = pd.concat([train_set,test_set])                      # concat 중복되는 값 합치기.
data['Date'] = pd.to_datetime(data['Date'])                 # type을 datetime으로 변경. datetime도 문자.

data['year'] = data['Date'].dt.strftime('%Y')               # data에서 년을 year로 지정. 
data['month'] = data['Date'].dt.strftime('%m')              # data에서 년을 month로 지정.
data['day'] = data['Date'].dt.strftime('%d')                # data에서 년을 day로 지정.
print(data)
print(data.shape)                   # (6435, 15)
print('=================')

print(test_set)
print(test_set.shape)                # (180, 11)

data = data.drop(['Date'], axis=1)                          # Date 값을 드롭.
data = data.fillna(0)                                       # data에 결측지를 0값으로 변환.
print(data)                          #[6435 rows x 14 columns]             

train_set = train_set.drop(['Temperature'], axis=1)
test_set = test_set.drop(['Temperature'],axis=1)

cols = ['IsHoliday','year','month','day']
for col in cols:
    le = LabelEncoder()
    data[col]=le.fit_transform(data[col])
train_set = data[0:len(train_set)]                          # 처음부터 train_set 까지 
test_set = data[len(train_set):]                            # train_set부터 마지막까지
 
# train_set[col]=le.fit_transform(train_set[col])
# test_set[col]=le.fit_transform(test_set[col])
print(train_set.isnull().sum())
print(train_set.shape) #(6255, 14)
print(test_set.shape) #(180, 14)

# print(train_set.columns)

# # ['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
# #       'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
# #       'IsHoliday', 'Weekly_Sales']
# print(train_set.shape)   # (6255, 12)

# print(train_set.info())  

# # Int64Index: 6255 entries, 1 to 6255
# # Data columns (total 12 columns):
# #  #   Column        Non-Null Count  Dtype
# # ---  ------        --------------  -----
# #  0   Store         6255 non-null   int64
# #  1   Date          6255 non-null   object
# #  2   Temperature   6255 non-null   float64
# #  3   Fuel_Price    6255 non-null   float64
# #  4   Promotion1    2102 non-null   float64
# #  5   Promotion2    1592 non-null   float64
# #  6   Promotion3    1885 non-null   float64
# #  7   Promotion4    1819 non-null   float64
# #  8   Promotion5    2115 non-null   float64
# #  9   Unemployment  6255 non-null   float64
# #  10  IsHoliday     6255 non-null   bool
# #  11  Weekly_Sales  6255 non-null   float64
# # dtypes: bool(1), float64(9), int64(1), object(1)     

# print(train_set.describe())  
# #              Store  Temperature  ...  Unemployment  Weekly_Sales
# # count  6255.000000  6255.000000  ...   6255.000000  6.255000e+03 
# # mean     23.000000    60.639199  ...      8.029236  1.047619e+06 
# # std      12.988211    18.624094  ...      1.874875  5.654362e+05 
# # min       1.000000    -2.060000  ...      4.077000  2.099862e+05 
# # 25%      12.000000    47.170000  ...      6.916500  5.538695e+05 
# # 50%      23.000000    62.720000  ...      7.906000  9.604761e+05 
# # 75%      34.000000    75.220000  ...      8.622000  1.421209e+06 
# # max      45.000000   100.140000  ...     14.313000  3.818686e+06 

# # [8 rows x 10 columns]

# print(train_set.isnull().sum())

# # Store              0
# # Date               0
# # Temperature        0
# # Fuel_Price         0
# # Promotion1      4153
# # Promotion2      4663
# # Promotion3      4370
# # Promotion4      4436
# # Promotion5      4140
# # Unemployment       0
# # IsHoliday          0
# # Weekly_Sales       0
# # dtype: int64

# train_set = train_set.fillna(0)       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
# test_set = test_set.fillna(0) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값
# print(train_set.isnull().sum()) 
# print(train_set.shape)   # (6255, 12)

# train_set['Date'] = pd.to_datetime(train_set['Date'])
# train_set['year'] = train_set['Date'].dt.year 
# train_set['month'] = train_set['Date'].dt.month
# train_set['day'] = train_set['Date'].dt.day
# train_set.drop(['Date'], inplace=True, axis=1)
# train_set['month'] = train_set['month']#.astype('category')

# test_set['Date'] = pd.to_datetime(test_set['Date'])
# test_set['year'] = test_set['Date'].dt.year 
# test_set['month'] = test_set['Date'].dt.month
# test_set['day'] = test_set['Date'].dt.day
# test_set.drop(['Date'], inplace=True, axis=1)
# test_set['month'] = test_set['month']#.astype('category')

# # int 정수 / float 실수 / object 문자+숫자 

# # train_set = train_set.drop(['Temperature'], axis=1)
# # test_set = test_set.drop(['Temperature'],axis=1)
# # train_set = train_set.drop(['Fuel_Price'], axis=1)
# # test_set = test_set.drop(['Fuel_Price'],axis=1)
# # train_set = train_set.drop(['Unemployment'], axis=1)
# # test_set = test_set.drop(['Unemployment'],axis=1)


# #cols 
# cols = ['IsHoliday']
# for col in cols:
#     le = LabelEncoder()
#     train_set[col]=le.fit_transform(train_set[col])
#     test_set[col]=le.fit_transform(test_set[col])

# print(train_set)
# print(test_set)


# x = train_set.drop(['Weekly_Sales'], axis=1)

# # test_set = test_set.drop(['Date'], axis=1)


# # drop 데이터에서 ''사이 값 빼기
# print(x)

# #       Store        Date  ...  Unemployment  IsHoliday
# # id                       ...
# # 1         1  05/02/2010  ...         8.106      False
# # 2         1  12/02/2010  ...         8.106       True
# # 3         1  19/02/2010  ...         8.106      False
# # 4         1  26/02/2010  ...         8.106      False
# # 5         1  05/03/2010  ...         8.106      False
# # ...     ...         ...  ...           ...        ...
# # 6251     45  31/08/2012  ...         8.684      False
# # 6252     45  07/09/2012  ...         8.684       True
# # 6253     45  14/09/2012  ...         8.684      False
# # 6254     45  21/09/2012  ...         8.684      False
# # 6255     45  28/09/2012  ...         8.684      False

# # [6255 rows x 11 columns]
# print(x.columns)
# print(x.shape)  

# y = train_set['Weekly_Sales'] 
# print(y)
# print(y.shape) 

# print(train_set.shape)
x = train_set.drop(['Weekly_Sales'], axis=1) #axis는 컬럼 
test_set = test_set.drop(['Weekly_Sales'], axis=1) #axis는 컬럼  
y = train_set['Weekly_Sales']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )
# scaler = MaxAbsScaler()
scaler = StandardScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)

# print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
# print(np.max(x_train))      # 1
# print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
# print(np.max(x_test))

#2. 모델구성
model = Sequential()
model.add(Dense(6400, activation='relu', input_dim=13))
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

# input1 = Input(shape=(9,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(128)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(64, activation='relu')(dense1)
# dense3 = Dense(32, activation='relu')(dense2)
# dense4 = Dense(16, activation='relu')(dense3)
# dense5 = Dense(8, activation='relu')(dense4)
# dense6 = Dense(4, activation='relu')(dense5)
# output1 = Dense(1, activation='relu')(dense6)

# model = Model(inputs = input1, outputs = output1)

import time
start_time = time.time()

#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='auto', verbose=1, 
                              restore_best_weights=True)
 

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=2000, batch_size=256, 
          verbose=1, validation_split=0.3, callbacks=[earlyStopping])

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)

# def RMSE(a, b):                     # 제곱값을 루트 적용. 
#     return np.sqrt(mean_squared_error(a, b))
# rmse = RMSE(y_test, y_predict)
def RMSE(y_test, y_predict):
     return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
# print(test_set.info())
# print(test_set.shape)
print(test_set.shape)
print(train_set.shape)


# test_set = test_set.astype
y_summit = model.predict(test_set)
# print(y_summit)
# print(y_summit.shape) 


sample_submission = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(sample_submission)

sample_submission['Weekly_Sales'] = y_summit
print(sample_submission)

sample_submission.to_csv('./_data/shopping/submission5.csv', index = True)

# RMSE :  511500.0304507484
# r2스코어 :  0.18623766920020768

# RMSE :  264422.55360759475
# r2스코어 :  0.7757539631822252

