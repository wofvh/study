import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import accuracy_score 
#1. 데이터


datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=100
                                                    )
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model =  RandomForestClassifier()

#3.4 컴파일,훈련, 예측
scores = cross_val_score(model,x_train, y_train,cv=kfold)
print('acc :' ,scores,'\n cross_val_score',round(np.mean(scores),4))

y_predict = cross_val_predict(model,x_test, y_test,cv=kfold)
acc =accuracy_score(y_test,y_predict)
print('cross_val_predict acc :', acc )