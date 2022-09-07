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
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,LSTM, Conv1D
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
model.add(Conv1D(128, 2, input_shape=(7,1)))
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))  



#3. 컴파일
# from tensorflow.keras.optimizers import Adam
# learing_rate = 0.01
# optimizer = Adam(learing_rate=learing_rate)

optimizer='adam'
model.compile(optimizer=optimizer,metrics=['acc'],
                loss='categorical_crossentropy')
    

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=50,mode='auto',verbose=1,factor=0.5)

start = time.time()
model.fit(x_train,y_train, epochs=600, batch_size=128,validation_split=0.2,callbacks=[es,reduced_lr])
end = time.time()-start

loss,acc = model.evaluate(x_test,y_test)

# print('model.score:',model.score) 
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
# y_predict = np.argmax(model.predict(x_test),axis=1)
# y_test =np.argmax(y_test)
print('걸린시간',end)
print('loss',loss)
print('acc',acc)


# 걸린시간 7.736591339111328
# loss nan
# acc 0.6143497824668884