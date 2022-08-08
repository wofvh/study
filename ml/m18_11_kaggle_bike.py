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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성 
parameters = [
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__min_samples_leaf':[3,5,7],'RF__min_samples_split':[2,3,5,20]},
    {'RF__min_samples_split':[2,3,5,20]},
    {'RF__n_jobs':[-1,2,4],'RF__min_samples_leaf':[3,5,7]}
]                

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kFold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestRegressor())],verbose=1)


#3. 컴파일 훈련
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

model =  RandomizedSearchCV(pipe ,parameters, cv =kFold ,verbose= 1)


import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
#4. 평가 예측
result = model.score(x_test, y_test)

print('model.score :',model.score(x_test,y_test))

y_predict= model.predict(x_test)
print('r2_score:',r2_score(y_test,y_predict))

print("걸린시간 :",round(end_time-start_time,4),"초")

# RandomizedSearchCV + KFold + pipe
# model.score : 0.9476483279017598
# r2_score: 0.9476483279017598
# 걸린시간 : 56.7329 초

# nopipeline 
# model.score : 0.9470482264623882
# r2_score: 0.9470482264623882
# 걸린시간 : 1.9187 초

# pipeline
# model.score : 0.9464769134267822
# r2_score: 0.9464769134267822
# 걸린시간 : 1.9198 초


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
