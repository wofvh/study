from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

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
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
path = './_data/house/' # ".은 현재 폴더"
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


#2. 모델구성

model2 = LinearRegression()
model3 = KNeighborsRegressor()
model4 = DecisionTreeRegressor()
model5 = RandomForestRegressor()

#3. 컴파일,훈련
# model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)

#4. 평가, 예측

# results1 = model1.score(x_test,y_test)   # = evaluate 
# print("Perceptron :",results1)     
# y_predict1 = model1.predict(x_test)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict1)
# print('r2 스코어 :', r2)

results2 = model2.score(x_test,y_test)   # = evaluate 
print("LinearRegression :",results2)  
y_predict2 = model2.predict(x_test)

results3 = model3.score(x_test,y_test)   # = evaluate 
print("KNeighborsRegressor :",results3)  
y_predict3 = model3.predict(x_test)

results4 = model4.score(x_test,y_test)   # = evaluate 
print("DecisionTreeRegressor :",results4)  
y_predict4 = model4.predict(x_test)

results5 = model5.score(x_test,y_test)   # = evaluate 
print("RandomForestRegressor :",results5)                 # 회귀는 = r2스코어 분류는 acc 값과 동일. 
y_predict5 = model5.predict(x_test)

# LogisticRegression()
# acc스코어 :  1.0

# LinearRegression : 0.8346514306091896
# KNeighborsRegressor : 0.7343487230294254       
# DecisionTreeRegressor : 0.7045800868229366
# RandomForestRegressor : 0.8899336467938522