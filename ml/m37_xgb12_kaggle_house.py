from sklearn.datasets import load_breast_cancer ,fetch_covtype, load_digits
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
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


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=123 )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = KFold(n_splits=n_splits ,shuffle=True, random_state=123)

# 'n_estimators':[100, 200, 300, 400, 500, 1000]}                 # 디폴트 100 /1-무한대/ 정수
# 'learning_rate':[0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001]            # 디폴트 0.3/ 0-1 / 다른이름:eta
# 'max_depth':[None, 2, 3, 4, 5, 6, 7, 8, 9, 10]                  # 디폴트 6  /0-무한대 /낮을수록 좋은편./정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10 ,100]                         # 디폴트 0  / 0-무한대 / # loss값을 조각낸다.
# 'min_child_weight':[0,0.1,0.001,1,2,3,4,5,6,10,100]             # 디폴트 1 / 0-무한대 
# 'subsample':[0,0.1,0.2,0.3,0.5,0.7,1]                           # 디폴트 1 / 0-1 데이터에 일정량을 샘플로 쓰겠다.
# 'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1]}                   # 디폴트 1 / 0-1
#'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1]}                   # 디폴트 1 / 0-1
#'colsample_byload':[0,0.1,0.2,0.3,0.5,0.7,1]}                    # 디폴트 1 / 0-1
# 'reg_alpha':[0,0.1,0.01,0.001,1,2,10]                           # 디폴트 0 / 0-무한대 / L1 절대값 가중치 규제 / alpha
# 'reg_lambda':[0,0.1,0.01,0.001,1,2,10]                          # 디폴트 1 / 0-무한대 / L2 제곱값 가중치 규제 / lambda

parameters = {'n_estimators':[1000],
              'learning_rate':[0.1],
              'max_depth':[3],
              'gamma': [1],
              'min_child_weight':[1],
              'subsample':[1],
              'colsample_bytree':[1],
              'colsample_bylevel':[1],
            #   'colsample_byload':[1],
              'reg_alpha':[0],
              'reg_lambda':[1]
              }  


#2. 모델 

xgb = XGBRegressor(random_state=123,
                    )

model = GridSearchCV(xgb, parameters, cv =kfold, n_jobs=8)

import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

print('최상의매개변수 : ',model.best_params_)
print('최상의 점수 :', model.best_score_)

#4. 평가 예측

results= model.score(x_test,y_test)
print("결과 :",results)
print("시간 :", end-start )

# 최상의매개변수 :  {'colsample_bylevel': 1, 'colsample_bytree': 1, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1}       
# 최상의 점수 : 0.869188426068632
# 결과 : 0.9104424515303553
# 시간 : 4.070497512817383
