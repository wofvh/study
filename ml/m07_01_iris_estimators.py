import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from yaml import warnings
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#                                                     train_size=0.8,
#                                                     random_state=66
#                                                     )

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
    

#3.4 컴파일,훈련, 예측

# 모델개수: 41
# cross_val_score : [0.96666667 0.96666667 1.         0.9        0.93333333]
# cross_val_score : [0.96666667 1.         1.         0.93333333 0.9       ]
# cross_val_score : [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333]
# cross_val_score : [0.93333333 0.96666667 1.         0.83333333 0.9       ]
# cross_val_score : [0.9        0.96666667 0.96666667 0.86666667 0.93333333]
# ClassifierChain 은 안나온 놈
# cross_val_score : [0.66666667 0.66666667 0.66666667 0.66666667 0.66666667]
# cross_val_score : [0.96666667 1.         1.         0.9        0.93333333]
# cross_val_score : [0.33333333 0.33333333 0.33333333 0.33333333 0.33333333]
# cross_val_score : [0.96666667 0.9        1.         0.9        0.93333333]
# cross_val_score : [0.96666667 0.96666667 1.         0.93333333 0.9       ]
# cross_val_score : [0.93333333 0.96666667 1.         0.93333333 0.9       ]
# cross_val_score : [0.96666667 0.96666667 1.         0.9        0.93333333]
# cross_val_score : [0.96666667 1.         1.         0.9        0.93333333]
# cross_val_score : [0.96666667 0.96666667 1.         0.93333333 0.93333333]
# cross_val_score : [0.96666667 0.96666667 1.         1.         0.93333333]
# cross_val_score : [0.96666667 0.93333333 1.         0.96666667 0.93333333]
# cross_val_score : [0.96666667 0.93333333 1.         0.96666667 0.93333333]
# cross_val_score : [1.  1.  1.  1.  0.9]
# cross_val_score : [1.         1.         1.         0.93333333 0.9       ]
# cross_val_score : [0.96666667 1.         1.         0.93333333 0.93333333]
# cross_val_score : [0.96666667 1.         1.         1.         0.93333333]
# cross_val_score : [1.         1.         1.         0.96666667 0.9       ]
# MultiOutputClassifier 은 안나온 놈
# cross_val_score : [0.96666667 0.96666667 1.         0.93333333 0.9       ]
# cross_val_score : [0.9        0.93333333 1.         0.83333333 0.96666667]
# cross_val_score : [0.96666667 0.96666667 1.         0.9        0.93333333]
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# cross_val_score : [0.9        0.93333333 0.66666667 0.83333333 0.86666667]
# cross_val_score : [0.96666667 0.8        0.96666667 0.7        0.8       ]
# cross_val_score : [1.         1.         1.         0.96666667 0.9       ]
# cross_val_score : [0.93333333 0.96666667 1.         0.86666667 0.96666667]
# cross_val_score : [0.96666667 0.96666667 1.         0.93333333 0.9       ]
# cross_val_score : [0.8        0.9        0.83333333 0.8        0.8       ]
# cross_val_score : [0.8        0.9        0.83333333 0.8        0.8       ]
# cross_val_score : [0.96666667 0.76666667 1.         0.83333333 0.83333333]
# cross_val_score : [0.96666667 0.96666667 1.         0.9        0.96666667]
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈