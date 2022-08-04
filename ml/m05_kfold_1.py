import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score

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
datasets = load_iris()
x = datasets['data']
y = datasets['target']

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#                                                     train_size=0.8,
#                                                     random_state=66
#                                                     )

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model = SVC()

#3.4 컴파일,훈련, 예측
scores = cross_val_score(model,x,y,cv=kfold)
# scores = cross_val_score(model,x,y,cv=10)  < 이것도 가능. 

print('acc :' ,scores,'\n cross_val_score',round(np.mean(scores),4))

# acc : [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score 0.9667