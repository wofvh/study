import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = [
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__min_samples_leaf':[3,5,7],'RF__min_samples_split':[2,3,5,20]},
    {'RF__min_samples_split':[2,3,5,20]},
    {'RF__n_jobs':[-1,2,4],'RF__min_samples_leaf':[3,5,7]}
]                

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=1)


#3. 컴파일 훈련
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

model = RandomizedSearchCV(pipe ,parameters, cv =5 ,verbose= 1)


import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
#4. 평가 예측
result = model.score(x_test, y_test)

print('model.score :',model.score(x_test,y_test))

from sklearn.metrics import accuracy_score
y_predict= model.predict(x_test)
print('acc_score:',accuracy_score(y_test,y_predict))

print("걸린시간 :",round(end_time-start_time,4),"초")

# GridSearchCV + KFold + pipe
# model.score : 0.9722222222222222
# acc_score: 0.9722222222222222
# 걸린시간 : 6.3725 초



# nopipeline 
# model.score : 0.9722222222222222
# acc_score: 0.9722222222222222
# 걸린시간 : 0.0821 초
# pipeline
# model.score : 0.9722222222222222
# acc_score: 0.9722222222222222
# 걸린시간 : 0.0816 초

# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=3)
# 최적의 파라미터: {'max_depth': 8, 'min_samples_leaf': 3}
# 최적의 점수: 0.9859605911330049
# model.score : 0.9722222222222222
# acc_score: 0.9722222222222222
# 최적의 튠 acc: 0.9722222222222222
# 걸린시간 : 27.437 

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestClassifier(max_depth=10, min_samples_leaf=3)
# 최적의 파라미터: {'n_estimators': 100, 'min_samples_leaf': 3, 'max_depth': 10}
# 최적의 점수: 0.9928571428571429
# model.score : 0.9722222222222222
# acc_score: 0.9722222222222222
# 최적의 튠 acc: 0.9722222222222222
# 걸린시간 : 2.9678 초
