# 과제
# ativation : sigmoid, relu, linear
# metrics 추가 
# earlystopping 포함.
# 성능비교
# 감상문 2줄이상 
# 구글원격 
# r2값? loss값 ? accuracy값? 
# california , diabet, boston >> 회귀모델 metrics=mse, mae 값 프린트 (relu 1.2,3 사용할 때마다 뭐가 다른지)


import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
# 과제
# ativation : sigmoid, relu, linear
# metrics 추가 
# earlystopping 포함.
# 성능비교
# 감상문 2줄이상 
# 구글원격 
# r2값? loss값 ? accuracy값? 
# california , diabet, boston >> 회귀모델 metrics=mse, mae 값 프린트 (relu 1.2,3 사용할 때마다 뭐가 다른지)


import numpy as np
from sklearn import datasets  
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR

#1. 데이터

datasets = fetch_california_housing()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


#2. 모델구성

model = LinearSVR()

#3 컴파일, 훈련
model.fit(x_train,y_train)

#4 평가 예측

results = model.score(x_test,y_test)   # = evaluate 
print("결과 :",results)                 # 회귀는 = r2스코어 분류는 acc 값과 동일. 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)

print('r2 스코어 :', r2)





datasets = load_boston()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


#2. 모델구성

model2 = LinearRegression()
model3 = KNeighborsRegressor()
model4 = DecisionTreeRegressor()
model5 = RandomForestRegressor()

#3. 컴파일,훈련
# model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
model5.fit(x_train,y_train)

#4. 평가, 예측

# results1 = model1.score(x_test,y_test)   # = evaluate 
# print("Perceptron :",results1)     
# y_predict1 = model1.predict(x_test)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict1)
# print('r2 스코어 :', r2)

results2 = model2.score(x_test,y_test)   # = evaluate 
print("LinearRegression :",results2)  
y_predict2 = model2.predict(x_test)

results3 = model3.score(x_test,y_test)   # = evaluate 
print("KNeighborsRegressor :",results3)  
y_predict3 = model3.predict(x_test)

results4 = model4.score(x_test,y_test)   # = evaluate 
print("DecisionTreeRegressor :",results4)  
y_predict4 = model4.predict(x_test)

results5 = model5.score(x_test,y_test)   # = evaluate 
print("RandomForestRegressor :",results5)                 # 회귀는 = r2스코어 분류는 acc 값과 동일. 
y_predict5 = model5.predict(x_test)
# model = LinearSVR()
# 결과 : 0.6481406476359292
# r2 스코어 : 0.6481406476359292

# LinearRegression : 0.7579692443889179
# KNeighborsRegressor : 0.5173517279063122       
# DecisionTreeRegressor : 0.8531298945020389     
# RandomForestRegressor : 0.9257799236249072   

