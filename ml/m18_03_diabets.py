from matplotlib.colors import rgb2hex
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import r2_score
#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)

parameters = [
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__min_samples_leaf':[3,5,7],'RF__min_samples_split':[2,3,5,20]},
    {'RF__min_samples_split':[2,3,5,20]},
    {'RF__n_jobs':[-1,2,4],'RF__min_samples_leaf':[3,5,7]}
]                

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

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
# model.score : 0.4387507321241968
# r2_score: 0.4387507321241967
# 걸린시간 : 8.6721 초


# nopipeline 
# model.score : 0.38112221135821833
# r2_score: 0.38112221135821833
# 걸린시간 : 0.1654 초
# pipeline
# model.score : 0.3950646371947487
# r2_score: 0.3950646371947487
# 걸린시간 : 0.1406 초

# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=7)
# 최적의 파라미터: {'max_depth': 12, 'min_samples_leaf': 7, 'n_estimators': 100}
# 최적의 점수: 0.44946392640816335
# model.score : 0.4742295050570766
# r2_score: 0.4742295050570766
# 최적의 튠 acc: 0.4742295050570766
# 걸린시간 : 13.0921 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestRegressor(min_samples_leaf=5, min_samples_split=3)
# 최적의 파라미터: {'min_samples_split': 3, 'min_samples_leaf': 5}
# 최적의 점수: 0.447254646865859
# model.score : 0.4611976873278393
# r2_score: 0.4611976873278393
# 최적의 튠 acc: 0.4611976873278393
# 걸린시간 : 3.1658 초