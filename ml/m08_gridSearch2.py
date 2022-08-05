import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
 
#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
n_splits =5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1,10,100,1000],"kernel":["linear"],"degree":[3,4,5]},     #12
    {"C":[1,10,100],"kernel":["rbf"],"gamma":[0.001,0.0001]},       #6
    {"C":[1,10,100,1000],"kernel":["sigmoid"],                      #24
     "gamma":[0.01,0.001,0.0001],"degree":[3,4]}
]                                                                   #총42 번
    

#2. 모델
model= SVC(C=1, kernel='linear', degree=3)
# model =GridSearchCV(SVC(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
#                     refit=True,n_jobs=-1)


#3. 컴파일,훈련
model.fit(x_train,y_train)

# print('최적의 매개변수 :',model.best_estimator_)
# # 최적의 매개변수 : SVC(C=1, kernel='linear')
# print("최적의 파라미터:",model.best_params_)
# # 최적의 파라미터: {'C': 1, 'degree': 3, 'kernel': 'linear'}
# print("최적의 점수:",model.best_score_)
# # 최적의 파라미터: 0.9916666666666668
print('model.score :',model.score(x_test,y_test))
# model.score : 0.9666666666666667

y_predict= model.predict(x_test)
print('acc_score:',accuracy_score(y_test,y_predict))
# acc_score: 0.9666666666666667

# y_pred_best = model.best_estimator_.predict(x_test)
# print('최적의 튠 acc:',accuracy_score(y_test,y_pred_best))
# # 최적의 튠 acc: 0.9666666666666667

