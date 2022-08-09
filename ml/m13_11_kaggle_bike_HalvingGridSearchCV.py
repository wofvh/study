from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from keras.layers.recurrent import  SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.experimental import enable_halving_search_cv   # 실험버전사용할때 사용.
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingGridSearchCV
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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
n_splits =5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

parameters = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7]},
    {'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7]},
    {'min_samples_leaf':[3,5,7],'min_samples_split':[2,3,5,20]},
    {'min_samples_split':[2,3,5,20]},
    {'n_jobs':[-1,2,4],'min_samples_leaf':[3,5,7]}
]                                                   
    
from sklearn.model_selection import RandomizedSearchCV
#2. 모델
# model= SVC(C=1, kernel='linear', degree=3)
model =HalvingGridSearchCV(RandomForestRegressor(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
                    refit=True,n_jobs=-1)


#3. 컴파일,훈련
import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
print('최적의 매개변수 :',model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("최적의 점수:",model.best_score_)
print('model.score :',model.score(x_test,y_test))

y_predict= model.predict(x_test)
print('r2_score:',r2_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 acc:',r2_score(y_test,y_pred_best))
print("걸린시간 :",round(end_time-start_time,4),"초")


# 최적의 매개변수 : RandomForestRegressor()
# 최적의 파라미터: {'min_samples_split': 2}
# 최적의 점수: 0.9470312481475084
# model.score : 0.9469746161144094
# r2_score: 0.9469746161144094
# 최적의 튠 acc: 0.9469746161144094
# 걸린시간 : 113.5078 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestRegressor(min_samples_split=3)
# 최적의 파라미터: {'min_samples_split': 3}
# 최적의 점수: 0.9468168448466117
# model.score : 0.9470869643097212
# r2_score: 0.9470869643097212
# 최적의 튠 acc: 0.9470869643097212
# 걸린시간 : 13.2219 초

# HalvingGridSearchCV
# 최적의 매개변수 : RandomForestRegressor()
# 최적의 파라미터: {'min_samples_split': 2}
# 최적의 점수: 0.9468603459831879
# model.score : 0.9479862499188586
# r2_score: 0.9479862499188586
# 최적의 튠 acc: 0.9479862499188586
# 걸린시간 : 68.855 초