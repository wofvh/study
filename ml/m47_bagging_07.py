# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 

from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import math
import numpy as np
import pandas as pd
#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
x = np.array(x)
x = np.delete(x,[4,6], axis=1)
y = np.array(y).reshape(-1, 1)



x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 
from sklearn.ensemble import BaggingClassifier,BaggingRegressor  # 한가지 모델을 여러번 돌리는 것(파라미터 조절).
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

model1 = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model2 = BaggingClassifier(RandomForestClassifier(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model3 = BaggingClassifier(LogisticRegression(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model4 = BaggingClassifier(XGBClassifier(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )


# model1 = DecisionTreeClassifier()
# model2 = RandomForestClassifier()
# model3 = GradientBoostingClassifier()
# model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
result1 = model1.score(x_test,y_test)
# print("model1.score:",result1)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model1.predict(x_test)
acc1 = accuracy_score(y_test,y_predict)

print( 'score1 :',acc1)
print(model1) 
print("===================================")

result2 = model2.score(x_test,y_test)
# print("model2.score:",result2)


y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test,y_predict2)

print( 'score2 :',acc2)
print(model2) 
print("===================================")

result3 = model3.score(x_test,y_test)
# print("model3.score3:",result3)


y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test,y_predict3)

print( 'score3 :',acc3)
print(model3)
print("===================================")

result4 = model4.score(x_test,y_test)
# print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test,y_predict4)

print( 'acc :',acc4)
print(model4) 
print("===================================")

# BaggingClassifier

# model1.score: 0.8491620111731844
# score1 : 0.8491620111731844
# BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100,
#                   n_jobs=1, random_state=123)
# ===================================
# model2.score: 0.8603351955307262
# score2 : 0.8603351955307262
# BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=100,
#                   n_jobs=1, random_state=123)
# ===================================
# model3.score3: 0.7653631284916201
# score3 : 0.7653631284916201
# BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100,
#                   n_jobs=1, random_state=123)
# ===================================
# model4.score: 0.8715083798882681
# acc : 0.8715083798882681
# BaggingClassifier(base_estimator=XGBClassifier


# model1.score: 0.9444444444444444
# score1 : 0.9444444444444444
# BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100,
#                   n_jobs=1, random_state=123)
# ===================================
# model2.score: 0.9777777777777777
# score2 : 0.9777777777777777
# BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=100,
#                   n_jobs=1, random_state=123)
# ===================================
# model3.score3: 0.9694444444444444
# score3 : 0.9694444444444444
# BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100,
#                   n_jobs=1, random_state=123)
# ===================================
# model4.score: 0.9694444444444444
# acc : 0.9694444444444444
# BaggingClassifier(base_estimator=XGBClassifier
# 삭제후 
# model.score: 0.8268156424581006
# accuracy_score : 0.8268156424581006
# DecisionTreeClassifier() : [0.10301101 0.33110716 0.23788123 0.0495797  0.2784209 ]
# ===================================
# model2.score: 0.8603351955307262
# accuracy2_score : 0.8603351955307262
# RandomForestClassifier() : [0.08560709 0.28300849 0.27950585 0.04720047 0.3046781 ]
# ===================================
# model3.score: 0.8435754189944135
# accuracy3_score : 0.8435754189944135
# GradientBoostingClassifier() : [0.1436283  0.49550638 0.15074869 0.04155378 0.16856285]     
# ===================================
# model4.score: 0.8603351955307262
# accuracy4_score : 0.8603351955307262
# XGBClassifier


# 삭제전 
# model.score: 0.8212290502793296
# accuracy_score : 0.8212290502793296
# DecisionTreeClassifier() : [0.0993953  0.32988035 0.22231528 0.04055697 0.03727911 0.25811193 
#  0.01246107]
# ===================================
# model2.score: 0.8491620111731844
# accuracy2_score : 0.8491620111731844
# RandomForestClassifier() : [0.08686047 0.26535174 0.26206507 0.05051479 0.03871302 0.26611846 
#  0.03037646]
# ===================================
# model3.score: 0.8603351955307262
# accuracy3_score : 0.8603351955307262
# GradientBoostingClassifier() : [0.14059592 0.4890627  0.14394682 0.04110958 0.00358098 0.1653559
#  0.0163481 ]
# ===================================
# model4.score: 0.8603351955307262
# accuracy4_score : 0.8603351955307262
# XGBClassifier




