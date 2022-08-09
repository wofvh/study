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
    

#2. 모델
# model= SVC(C=1, kernel='linear', degree=3)
model =GridSearchCV(RandomForestRegressor(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
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

# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=7)
# 최적의 파라미터: {'max_depth': 12, 'min_samples_leaf': 7, 'n_estimators': 100}
# 최적의 점수: 0.44946392640816335
# model.score : 0.4742295050570766
# r2_score: 0.4742295050570766
# 최적의 튠 acc: 0.4742295050570766
# 걸린시간 : 13.0921 초