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

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

#2. 모델
model1 = Perceptron()
model2 = LogisticRegression()
model3 = KNeighborsClassifier()
model4 = DecisionTreeClassifier()
model5 = RandomForestClassifier()

#3. 컴파일,훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)

#4. 평가, 예측

results1 = model1.score(x_test,y_test)   # = evaluate 
print("Perceptron :",results1)     

results2 = model2.score(x_test,y_test)   # = evaluate 
print("LogisticRegression :",results2)  

results3 = model3.score(x_test,y_test)   # = evaluate 
print("KNeighborsClassifier :",results3)  

results4 = model4.score(x_test,y_test)   # = evaluate 
print("DecisionTreeClassifier :",results4)  

results5 = model5.score(x_test,y_test)   # = evaluate 
print("RandomForestClassifier :",results5)  

#4. 평가, 예측


# loss :  0.0530550517141819
# accuracy :  1.0

# model = LinearSVC()
# 결과 : 0.9472222222222222
# acc스코어 :  0.9472222222222222

# Perceptron : 0.9388888888888889        
# LogisticRegression : 0.9722222222222222
# KNeighborsClassifier : 0.9916666666666667  
# DecisionTreeClassifier : 0.8333333333333334
# RandomForestClassifier : 0.9777777777777777