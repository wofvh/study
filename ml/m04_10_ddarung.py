from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 # pandas : 엑셀땡겨올때 씀 python 지원하는 엑셀을 불러오는 기능.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)

train_set = train_set.fillna(train_set.mean())       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값

x = train_set.drop(['count'], axis=1)                    # drop 데이터에서 ''사이 값 빼기

y = train_set['count'] 

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
# 결과 : -0.11612894209723268
# r2 스코어 : -0.11612894209723268

# LinearRegression : 0.5809180909957794
# KNeighborsRegressor : 0.2991859652598958       
# DecisionTreeRegressor : 0.5426947005004573     
# RandomForestRegressor : 0.7554253145184169



