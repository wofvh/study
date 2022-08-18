import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston, fetch_california_housing
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from keras.layers.recurrent import  SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

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

print(train_set)# [10886 rows x 13 columns]
print(test_set)# [6493 rows x 12 columns]

##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)
y = train_set['count'] 
print(y)
print(y.shape) # (10886,)
# x = np.array(x)
# x = np.delete(x,[0,1,7], axis=1) 

# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


# print(x.shape,y.shape)
# print(datasets.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)



from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import  KFold , StratifiedKFold
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 .모델


lgb = LGBMRegressor()
cvb = CatBoostRegressor(verbose=0)
xg =XGBRegressor()


model = VotingRegressor(estimators=[('LGB',lgb),('CVB',cvb),('XG',xg)],)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가
y_predict = model.predict(x_test)

score = r2_score(y_test,y_predict)
print('voting결과',round(score,4))

# 결과 0.9912

classifiers = [lgb,cvb,xg]
for model2 in classifiers :
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name =model2.__class__.__name__
    print('{0} 정확도:{1:.4f}'.format(class_name,score2))
    
    
# voting결과 0.9565
# LGBMRegressor 정확도:0.9493
# CatBoostRegressor 정확도:0.9563
# XGBRegressor 정확도:0.9495


