from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import math
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,LSTM
#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')


print(train_set.describe())                             # 문자값을 뺀 결과 숫자만 나옴. 

# PassengerId    Survived  ...       Parch        Fare
# count   891.000000  891.000000  ...  891.000000  891.000000
# mean    446.000000    0.383838  ...    0.381594   32.204208
# std     257.353842    0.486592  ...    0.806057   49.693429
# min       1.000000    0.000000  ...    0.000000    0.000000
# 25%     223.500000    0.000000  ...    0.000000    7.910400
# 50%     446.000000    0.000000  ...    0.000000   14.454200
# 75%     668.500000    1.000000  ...    0.000000   31.000000
# max     891.000000    1.000000  ...    6.000000  512.329200
# [8 rows x 7 columns]

print(train_set.info())

# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.7+ KB

print(train_set.isnull())                       # isnull() = 값중에 nan값을 True로 표시해준다. 
# PassengerId  Survived  Pclass  ...   Fare  Cabin  Embarked
# 0          False     False   False  ...  False   True     False    
# 1          False     False   False  ...  False  False     False    
# 2          False     False   False  ...  False   True     False    
# 3          False     False   False  ...  False  False     False    
# 4          False     False   False  ...  False   True     False    
# ..           ...       ...     ...  ...    ...    ...       ...    
# 886        False     False   False  ...  False   True     False    
# 887        False     False   False  ...  False  False     False    
# 888        False     False   False  ...  False   True     False    
# 889        False     False   False  ...  False  False     False    
# 890        False     False   False  ...  False   True     False    

# [891 rows x 12 columns]

print(train_set.isnull().sum())                  # sum() = non값을 카운트해준다. 
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# dtype: int64

print(train_set.shape) # (891, 12)

print(train_set.columns.values)
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
# 'Ticket' 'Fare' 'Cabin' 'Embarked']


train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)  
# mean : 평균값 / inplace = True / fillna : 특정 값으로 결측치의 데이터를 채운다.
# fillna 는 새로운 객체를 반환하지만, 기존 객체를 변경할 수도 있다. 객체를 변경할 때 
# inplace=True를 설정.  
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
# print(train_set['Embarked'])  = 'S':0,'C':1,'Q':2
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set

y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.25,
                                                    random_state=58525
                                                    )

print(x_train.shape) # (712, 7)
print(y_train.shape) # (712, 1)
print(x_test.shape) # (179, 7)
print(y_test.shape) # (179, 1)
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(y)
print(y.shape) # (1459,)
print(x_train.shape) #(668, 7)
print(x_test.shape) #(223, 7)

x_train = x_train.reshape(668, 7,1)
x_test = x_test.reshape(223, 7,1)



# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)
# # scaler.transform(x_test)
# x_test =scaler.transform(x_test)
# x_train = scaler.transform(x_train)
import time

#2. 모델구성
model = Sequential()
model.add(LSTM(350, input_shape=(7,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(4, activation='swish'))
model.add(Dense(1))
model.summary()

# input1 = Input(shape=(7,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(80, activation='relu')(dense2)
# dense4 = Dense(15, activation='relu')(dense3)
# output1 = Dense(1, activation='sigmoid')(dense4)
# model = Model(inputs = input1, outputs = output1)



start_time = time.time()
#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 
filepath = './_ModelCheckPoint/12titanic/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
        
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
                      save_best_only=True,                                      # patience 필요없음.
                      filepath ="".join([filepath,'12titanic_',date, '_', filename]))

earlyStopping = EarlyStopping(monitor='val_loss', patience=80, mode='min', verbose=1, 
                              restore_best_weights=True)


model.fit(x_train, y_train, epochs=500, batch_size=3400, verbose=1, 
          validation_split=0.2, callbacks=[earlyStopping, mcp])
end_time = time.time() - start_time

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

acc1 = accuracy_score(y_test, y_predict) 
print('acc1 : ', acc1) 
print('걸린시간 :', end_time)

#############################################################################
# submission = pd.read_csv(path + 'submission.csv',index_col=0)
# test_set = pd.read_csv(path + 'gender_submission.csv', index_col=0)

# y_summit = model.predict(test_set)



# submission['count'] = y_summit 

# submission.to_csv(path + 'submission.csv', index=False)

# loss :  0.45621979236602783
# accuracy :  0.7937219738960266
# acc1 :  1.0
# 걸린시간 : 13.473679542541504

# LSTM
# loss :  0.6224689483642578
# accuracy :  0.6367713212966919
# acc1 :  1.0
# 걸린시간 : 25.49001717567444