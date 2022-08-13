from sklearn.datasets import load_breast_cancer ,fetch_covtype, load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time 
import numpy as np
import pandas as pd
#1. 데이터 
#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
x = np.array(x)
x = np.delete(x,[4,6], axis=1)
y = np.array(y).reshape(-1, 1)



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=123, stratify = y )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5

kfold = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=123)

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

xgb = XGBClassifier(random_state=123,
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
# 최상의 점수 : 0.8272234807446075
# 결과 : 0.8156424581005587
# 시간 : 2.2801730632781982
