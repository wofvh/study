from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 대문자 class  암시가능.
from sklearn.preprocessing import MaxAbsScaler, RobustScaler  
import matplotlib

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
path = './study/_data/house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


# print(test_set)
# print(train_set.shape) #(1460,76)
# print(test_set.shape) #(1459, 75) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

# print(train_set.columns)
# print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
# print(train_set.describe()) 

###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.89, shuffle = True, random_state = 68
 )

import time

scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
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
# model = Sequential()
# model.add(Dense(100,input_dim=75))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))
# model.summary()
# # Total params: 21,201
# # Trainable params: 21,201
# # Non-trainable params: 0

# input1 = Input(shape=(75,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(100)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(100, activation='relu')(dense2)
# output1 = Dense(1)(dense3)

# model = Model(inputs = input1, outputs = output1)

start_time = time.time()

3. #컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=80, mode='min', 
                              verbose=1,restore_best_weights=True)
# model.compile(loss='mae', optimizer='adam')
# hist = model.fit(x_train, y_train, epochs=500, batch_size=32, 
#                 validation_split=0.3,
#                 callbacks = [earlyStopping],
#                 verbose=1
#                )
# model.save("./_save/keras23_17_load_house.h5")
model = load_model("./_save/keras23_17_load_house.h5")
end_time = time.time() - start_time

#verbose = 0으로 할 시 출력해야할 데이터가 없어 속도가 빨라진다.강제 지연 발생을 막는다.
#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
print('걸린시간 :', end_time)

# loss : 16815.8046875
# r2스코어 : 0.8757784845378902
# 걸린시간 : 35.940136194229126

# save 후 
# loss : 16815.8046875
# r2스코어 : 0.8757784845378902
# 걸린시간 : 0.13663268089294434





