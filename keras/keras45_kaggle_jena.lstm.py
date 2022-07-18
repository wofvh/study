
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout ,Conv1D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import time
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/kaggle_jena/'

df = pd.read_csv(path + 'jena_climate_2009_2016.csv')                
                          

df = df.drop(['Date Time'], axis=1) 

print(df)
print(df.shape) # (420551, 14)

print(df.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    #    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    #    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',        
    #    'wd (deg)'],
    #   dtype='object')
    
print(df.info())          
#  #   Column           Non-Null Count   Dtype
# ---  ------           --------------   -----
#  0   p (mbar)         420551 non-null  float64
#  1   T (degC)         420551 non-null  float64
#  2   Tpot (K)         420551 non-null  float64
#  3   Tdew (degC)      420551 non-null  float64
#  4   rh (%)           420551 non-null  float64
#  5   VPmax (mbar)     420551 non-null  float64
#  6   VPact (mbar)     420551 non-null  float64
#  7   VPdef (mbar)     420551 non-null  float64
#  8   sh (g/kg)        420551 non-null  float64
#  9   H2OC (mmol/mol)  420551 non-null  float64
#  10  rho (g/m**3)     420551 non-null  float64
#  11  wv (m/s)         420551 non-null  float64
#  12  max. wv (m/s)    420551 non-null  float64
#  13  wd (deg)         420551 non-null  float64
# dtypes: float64(14)
# memory usage: 48.1+ MB
# None
print(df.describe())  
# None
#             p (mbar)       T (degC)  ...  max. wv (m/s)       wd (deg)
# count  420551.000000  420551.000000  ...  420551.000000  420551.000000        
# mean      989.212776       9.450147  ...       3.056555     174.743738        
# std         8.358481       8.423365  ...      69.016932      86.681693        
# min       913.600000     -23.010000  ...   -9999.000000       0.000000        
# 25%       984.200000       3.360000  ...       1.760000     124.900000        
# 50%       989.580000       9.420000  ...       2.960000     198.100000        
# 75%       994.720000      15.470000  ...       4.740000     234.100000        
# max      1015.350000      37.280000  ...      23.500000     360.000000        

# [8 rows x 14 columns]



a = df

size = 5 # x= 4개 y는 1개
def split_x(dataset, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa = []   #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset)- size + 1): # 6이다 range(횟수)
        subset = dataset[i : (i + size)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa.append(subset) #append 마지막에 요소를 추가한다는 뜻
    return np.array(aaa)    


bbb = split_x(a, size)


x = bbb[:,:-1]
y = bbb[:,-1]



print(x.shape) # (420547, 4, 14)
print(y.shape) # (420547, 14)
x = x.reshape(420547, 4, 14)
y = y.reshape(420547, 14 ,1)
print(x.shape) # (420547, 4, 14)
print(y.shape) # (96,1,1)
# print(z.shape) # (6, 4)
# print(z) # (6, 4)
 
# df =df(['Date Time'], axis=1)


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=66,shuffle=False)

print(x_train.shape) # (294382, 4, 14)
print(x_test.shape) # (126165, 4, 14)




#2. 모델

model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(LSTM(units= 1000, input_shape=(4,14)))
# model.add(Flatten())
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
 

                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()      # 회귀모델은 output = linear 자연수치 그데로 나와야 함. 디폴트.
                                                # * 분류모델은 이진 > 마지막 activation = sigmoid 

# input1 = Input(shape=(30,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(50, activation='relu')(dense1)
# dense3 = Dense(30, activation='sigmoid')(dense2)
# output1 = Dense(1)(dense3)

# model = Model(inputs = input1, outputs = output1)

import time

#3 컴파일, 훈련
model.compile(loss ='mae', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.
         

earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
              verbose=1, restore_best_weights = True)     
                                                                
hist = model.fit(x_train, y_train, epochs =100, batch_size = 3200, 
                 verbose=1, 
                 validation_split = 0.3,
                 callbacks = [earlystopping])      # callbacks으로 불러온다 erlystopping   

#4 평가 예측

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# loss :  195.19041442871094