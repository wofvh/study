# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


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
from sklearn.metrics import r2_score
#1. 데이터

datasets = load_boston()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)


x = np.delete(x,1, axis=1) 


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 
#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 
from sklearn.ensemble import BaggingClassifier ,BaggingRegressor # 한가지 모델을 여러번 돌리는 것(파라미터 조절).,
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

model1 = BaggingRegressor(DecisionTreeRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model2 = BaggingRegressor(RandomForestRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model3 = BaggingRegressor(KNeighborsRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model4 = BaggingRegressor(XGBRFRegressor(),
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
score1 = r2_score(y_test,y_predict)

print( 'score1 :',score1)
print(model1) 
print("===================================")

result2 = model2.score(x_test,y_test)
# print("model2.score:",result2)


y_predict2 = model2.predict(x_test)
score2 = r2_score(y_test,y_predict2)

print( 'score2 :',score2)
print(model2) 
print("===================================")

result3 = model3.score(x_test,y_test)
# print("model3.score3:",result3)


y_predict3 = model3.predict(x_test)
score3 = r2_score(y_test,y_predict3)

print( 'score3 :',score3)
print(model3)
print("===================================")

result4 = model4.score(x_test,y_test)
# print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
score4 = r2_score(y_test,y_predict4)

print( 'acc :',score4)
print(model4) 
print("===================================")

# BaggingClassifier
# score1 : 0.919348045497619
# BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score2 : 0.9177299012087415
# BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score3 : 0.8600341294048086
# BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# acc : 0.9176953476944002
# BaggingRegressor(base_estimator=XGBRFRegressor

# 삭제후 
# model.score: 0.82936488225
# r2_score1 : 0.82936488225
# DecisionTreeRegressor() :
# ===================================
# model1.score: 0.9294557702876841
# r2_score2 : 0.9294557702876841
# RandomForestRegressor() :
# ===================================
# model2.score3: 0.9273402196190076
# r2_score3 : 0.9273402196190076
# RandomForestRegressor() :
# ===================================
# model4.score: 0.920880286358798
# r2_score4 : 0.920880286358798
# XGBRFRegressor


# 삭제전 
# model.score: 0.8473492835526287
# r2_score1 : 0.8473492835526287
# DecisionTreeRegressor() :
# ===================================
# model1.score: 0.926309018874254
# r2_score2 : 0.926309018874254
# RandomForestRegressor() :
# ===================================
# model2.score3: 0.9238869646811706
# r2_score3 : 0.9238869646811706
# RandomForestRegressor() :
# ===================================
# model4.score: 0.9150650377981372
# r2_score4 : 0.9150650377981372
# XGBRFRegressor




