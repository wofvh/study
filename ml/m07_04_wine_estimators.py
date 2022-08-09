import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
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
#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']


from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model =  RandomForestClassifier()

from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') 
from sklearn.preprocessing import MinMaxScaler


#2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms:',allAlgorithms)
print('모델개수:',len(allAlgorithms))

for (name,algorithm) in  allAlgorithms :
    try :
        model = algorithm()

        scores = cross_val_score(model,x, y,cv=kfold)
        print('cross_val_score :' ,scores)
    except:
        # continue
        print(name,"은 안나온 놈")
        
# 모델개수: 41
# cross_val_score : [0.91666667 0.80555556 0.91666667 0.88571429 0.57142857]
# cross_val_score : [0.94444444 1.         1.         1.         0.91428571]
# cross_val_score : [0.38888889 0.38888889 0.38888889 0.4        0.42857143]
# cross_val_score : [0.88888889 1.         0.88888889 0.91428571 0.94285714]
# cross_val_score : [       nan        nan 0.91666667        nan        nan]
# ClassifierChain 은 안나온 놈
# cross_val_score : [0.47222222 0.66666667 0.72222222 0.68571429 0.77142857]
# cross_val_score : [0.88888889 0.94444444 0.83333333 1.         0.97142857]
# cross_val_score : [0.38888889 0.38888889 0.38888889 0.4        0.42857143]
# cross_val_score : [0.80555556 0.86111111 0.75       0.82857143 0.94285714]
# cross_val_score : [1.         0.94444444 1.         1.         1.        ]
# cross_val_score : [0.97222222 0.94444444 1.         0.97142857 0.94285714]
# cross_val_score : [0.52777778 0.38888889 0.5        0.54285714 0.45714286]
# cross_val_score : [0.94444444 0.97222222 0.94444444 1.         0.91428571]
# cross_val_score : [0.97222222 0.94444444 0.94444444 1.         0.97142857]
# cross_val_score : [0.55555556 0.61111111 0.69444444 0.74285714 0.8       ]
# cross_val_score : [0.44444444 0.5        0.55555556 0.54285714 0.45714286]
# cross_val_score : [0.44444444 0.5        0.55555556 0.54285714 0.45714286]
# cross_val_score : [0.97222222 1.         0.97222222 1.         1.        ]
# cross_val_score : [0.66666667 0.97222222 0.83333333 0.8        0.94285714]
# cross_val_score : [0.91666667 0.97222222 0.94444444 0.97142857 0.94285714]
# cross_val_score : [0.94444444 0.97222222 0.97222222 0.97142857 0.97142857]
# cross_val_score : [0.38888889 0.97222222 0.88888889 0.94285714 0.62857143]
# MultiOutputClassifier 은 안나온 놈
# cross_val_score : [0.72222222 0.91666667 0.83333333 0.88571429 0.91428571]
# cross_val_score : [0.61111111 0.77777778 0.75       0.77142857 0.74285714]
# cross_val_score : [0.83333333 0.91666667 0.86111111 0.97142857 0.85714286]
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# cross_val_score : [0.5        0.44444444 0.63888889 0.68571429 0.71428571]
# cross_val_score : [0.44444444 0.66666667 0.66666667 0.57142857 0.45714286]
# cross_val_score : [0.97222222 1.         1.         1.         1.        ]
# cross_val_score : [nan nan nan nan nan]
# cross_val_score : [1.         0.94444444 1.         1.         0.97142857]
# cross_val_score : [0.97222222 1.         1.         0.97142857 1.        ]
# cross_val_score : [0.97222222 1.         0.97222222 0.97142857 1.        ]
# cross_val_score : [0.5        0.61111111 0.63888889 0.51428571 0.74285714]
# cross_val_score : [0.55555556 0.77777778 0.72222222 0.71428571 0.68571429]
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
