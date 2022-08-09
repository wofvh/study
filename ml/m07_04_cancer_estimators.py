import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
datasets = load_breast_cancer()
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
# cross_val_score : [0.94736842 0.93859649 0.95614035 0.97368421 0.97345133]
# cross_val_score : [0.92982456 0.96491228 0.96491228 0.95614035 0.97345133]
# cross_val_score : [0.62280702 0.62280702 0.63157895 0.63157895 0.62831858]
# cross_val_score : [0.92105263 0.9122807  0.95614035 0.92982456 0.91150442]
# cross_val_score : [       nan        nan 0.96491228        nan 0.92920354]
# ClassifierChain 은 안나온 놈
# cross_val_score : [0.89473684 0.89473684 0.92982456 0.88596491 0.86725664]
# cross_val_score : [0.9122807  0.94736842 0.9122807  0.90350877 0.89380531]
# cross_val_score : [0.62280702 0.62280702 0.63157895 0.63157895 0.62831858]
# cross_val_score : [0.9122807  0.95614035 0.93859649 0.88596491 0.9380531 ]
# cross_val_score : [0.95614035 0.98245614 0.98245614 0.95614035 0.96460177]
# cross_val_score : [0.93859649 0.93859649 0.93859649 0.92105263 0.96460177]
# cross_val_score : [0.88596491 0.9122807  0.89473684 0.93859649 0.89380531]
# cross_val_score : [0.94736842 0.96491228 0.97368421 0.94736842 0.97345133]
# cross_val_score : [0.97368421 0.95614035 1.         0.92105263 0.96460177]
# cross_val_score : [0.90350877 0.92105263 0.93859649 0.95614035 0.92920354]
# cross_val_score : [0.42105263 0.38596491 0.38596491 0.38596491 0.38938053]
# cross_val_score : [0.42105263 0.38596491 0.38596491 0.38596491 0.38938053]
# cross_val_score : [0.95614035 0.95614035 0.97368421 0.96491228 0.94690265]
# cross_val_score : [0.9122807  0.92105263 0.84210526 0.92982456 0.87610619]
# cross_val_score : [0.92982456 0.92982456 0.96491228 0.94736842 0.95575221]
# cross_val_score : [0.96491228 0.93859649 0.95614035 0.95614035 0.94690265]
# cross_val_score : [0.92982456 0.90350877 0.95614035 0.92105263 0.91150442]
# MultiOutputClassifier 은 안나온 놈
# cross_val_score : [0.89473684 0.89473684 0.92982456 0.88596491 0.86725664]
# cross_val_score : [0.85964912 0.89473684 0.92982456 0.85964912 0.90265487]
# cross_val_score : [0.85964912 0.86842105 0.92982456 0.84210526 0.88495575]
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# cross_val_score : [0.92105263 0.87719298 0.81578947 0.92105263 0.90265487]
# cross_val_score : [0.89473684 0.87719298 0.72807018 0.70175439 0.89380531]
# cross_val_score : [0.97368421 0.97368421 0.94736842 0.94736842 0.9380531 ]
# cross_val_score : [nan nan nan nan nan]
# cross_val_score : [0.92105263 0.97368421 0.98245614 0.95614035 0.95575221]
# cross_val_score : [0.95614035 0.95614035 0.97368421 0.94736842 0.94690265]
# cross_val_score : [0.96491228 0.96491228 0.98245614 0.94736842 0.9380531 ]
# cross_val_score : [0.86842105 0.49122807 0.90350877 0.83333333 0.65486726]
# cross_val_score : [0.87719298 0.92982456 0.95614035 0.9122807  0.91150442]
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈