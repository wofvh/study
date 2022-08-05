import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
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
datasets = load_digits()
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
        
#         모델개수: 41
# cross_val_score : [0.25277778 0.26388889 0.24512535 0.25069638 0.33983287]
# cross_val_score : [0.92777778 0.96111111 0.91643454 0.9275766  0.94428969]
# cross_val_score : [0.81666667 0.89166667 0.82729805 0.83844011 0.86908078]
# cross_val_score : [0.96111111 0.97222222 0.95543175 0.94428969 0.96935933]
# cross_val_score : [nan nan nan nan nan]
# ClassifierChain 은 안나온 놈
# cross_val_score : [0.78611111 0.84722222 0.82172702 0.80222841 0.83286908]
# cross_val_score : [0.85555556 0.91388889 0.85236769 0.83008357 0.84958217]
# cross_val_score : [0.1        0.1        0.10027855 0.10306407 0.10306407]
# cross_val_score : [0.74166667 0.825      0.78551532 0.81615599 0.7632312 ]
# cross_val_score : [0.97777778 0.98333333 0.98050139 0.97493036 0.98885794]
# cross_val_score : [0.78888889 0.875      0.81615599 0.8189415  0.86629526]
# cross_val_score : [0.10833333 0.1        0.10306407 0.11142061 0.10584958]
# cross_val_score : [0.96111111 0.975      0.96100279 0.94428969 0.96100279]
# cross_val_score : [0.96388889 0.98611111 0.96935933 0.96935933 0.98050139]
# cross_val_score : [0.98888889 0.98888889 0.98328691 0.98050139 0.99164345]
# cross_val_score : [0.10277778 0.1        0.10027855 0.10027855 0.09749304]
# cross_val_score : [0.10277778 0.1        0.10027855 0.10027855 0.09749304]
# cross_val_score : [0.95277778 0.96111111 0.93871866 0.93593315 0.97214485]
# cross_val_score : [0.95       0.95555556 0.93593315 0.9275766  0.95264624]
# cross_val_score : [0.97777778 0.975      0.95264624 0.94428969 0.96657382]
# cross_val_score : [0.96111111 0.975      0.9637883  0.94986072 0.98328691]
# cross_val_score : [0.97777778 0.98333333 0.97493036 0.95543175 0.99164345]
# MultiOutputClassifier 은 안나온 놈
# cross_val_score : [0.88333333 0.91944444 0.88857939 0.88579387 0.91922006]
# cross_val_score : [0.89166667 0.91388889 0.87743733 0.88300836 0.92200557]
# cross_val_score : [0.95833333 0.96666667 0.95543175 0.94707521 0.96657382]
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# cross_val_score : [0.93333333 0.96666667 0.94707521 0.93593315 0.94707521]
# cross_val_score : [0.925      0.94166667 0.9275766  0.91364903 0.91364903]
# cross_val_score : [0.79444444 0.89166667 0.79108635 0.89415042 0.84122563]
# cross_val_score : [nan nan nan nan nan]
# cross_val_score : [0.96388889 0.98333333 0.95821727 0.97214485 0.98885794]
# cross_val_score : [0.94444444 0.95       0.93593315 0.91364903 0.94428969]
# cross_val_score : [0.94444444 0.95       0.93593315 0.91364903 0.94428969]
# cross_val_score : [0.91666667 0.96944444 0.93871866 0.94428969 0.9637883 ]
# cross_val_score : [0.98611111 0.99166667 0.99442897 0.98328691 0.99442897]
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈