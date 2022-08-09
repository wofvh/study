from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 # pandas : 엑셀땡겨올때 씀 python 지원하는 엑셀을 불러오는 기능.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)

train_set = train_set.fillna(train_set.mean())       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값

x = train_set.drop(['count'], axis=1)                    # drop 데이터에서 ''사이 값 빼기

y = train_set['count'] 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


#2. 모델구성

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model =  RandomForestRegressor ()

#3.4 컴파일,훈련, 예측
scores = cross_val_score(model,x_train, y_train,cv=kfold)
print('r2 :' ,scores,'\n cross_val_score',round(np.mean(scores),4))

y_predict = cross_val_predict(model,x_test, y_test,cv=kfold)
r2 =r2_score(y_test,y_predict)
print('cross_val_predict r2 :', r2 )

# r2 : [0.79708631 0.7995737  0.76954605 0.70698432 0.80090824] 
#  cross_val_score 0.7748
# cross_val_predict r2 : 0.6239997039963494

