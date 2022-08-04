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

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=58525
                                                    )

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model =  RandomForestRegressor ()

#3.4 컴파일,훈련, 예측
scores = cross_val_score(model,x_train, y_train,cv=kfold)
print('r2 :' ,scores,'\n cross_val_score',round(np.mean(scores),4))

y_predict = cross_val_predict(model,x_test, y_test,cv=kfold)
r2 =r2_score(y_test,y_predict)
print('cross_val_predict r2 :', r2 )

# acc : [0.48730533 0.55508641 0.3962664  0.55995718 0.41721051] 
#  cross_val_score 0.4832
# cross_val_predict acc : 0.27806914337779953