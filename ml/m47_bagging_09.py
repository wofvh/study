# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


import numpy as np
from sklearn import datasets  
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import r2_score

#1. 데이터

datasets = fetch_california_housing()

x = datasets.data                       #(569, 30)
y = datasets.target  

# x = np.array(x)
# y = np.array(y) 

# x = np.delete(x,1, axis=1) 
# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


# x = np.delete(x,1, axis=1) 


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = RandomForestRegressor()
model4 = XGBRFRegressor ()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
result1 = model1.score(x_test,y_test)
print("model.score:",result1)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model1.predict(x_test)
r2 = r2_score(y_test,y_predict)

print( 'r2_score1 :',r2)
print(model1,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result2 = model2.score(x_test,y_test)
print("model1.score:",result2)


y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test,y_predict2)

print( 'r2_score2 :',r2)
print(model2,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result3 = model3.score(x_test,y_test)
print("model2.score3:",result3)


y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test,y_predict3)

print( 'r2_score3 :',r2)
print(model3,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result4 = model4.score(x_test,y_test)
print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
r2 = r2_score(y_test,y_predict4)

print( 'r2_score4 :',r2)
print(model4,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")
# 삭제후 

# model.score: 0.6212201764048555
# r2_score1 : 0.6212201764048555
# DecisionTreeRegressor() :
# ===================================
# model1.score: 0.8102127271505424
# r2_score2 : 0.8102127271505424
# RandomForestRegressor() :
# ===================================
# model2.score3: 0.8108216155264999
# r2_score3 : 0.8108216155264999
# RandomForestRegressor() :
# ===================================
# model4.score: 0.6803213304653157
# r2_score4 : 0.6803213304653157
# XGBRFRegressor


# 삭제전 
# model.score: 0.6009084147892423
# r2_score1 : 0.6009084147892423
# DecisionTreeRegressor() :
# ===================================
# model1.score: 0.806518415840741
# r2_score2 : 0.806518415840741
# RandomForestRegressor() :
# ===================================
# model2.score3: 0.8047064408283028
# r2_score3 : 0.8047064408283028
# RandomForestRegressor() :
# ===================================
# model4.score: 0.694272779577398
# r2_score4 : 0.694272779577398
# XGBRFRegressor




