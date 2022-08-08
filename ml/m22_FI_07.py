# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sqlalchemy import column

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

# x = np.array(x)
# y = np.array(y) 

x = np.delete(x,1, axis=1) 
# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


print(x.shape,y.shape)
print(datasets.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

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
print(model1,':',model1.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result2 = model2.score(x_test,y_test)
print("model1.score:",result2)


y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test,y_predict2)

print( 'r2_score2 :',r2)
print(model2,':',model2.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result3 = model3.score(x_test,y_test)
print("model2.score3:",result3)


y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test,y_predict3)

print( 'r2_score3 :',r2)
print(model3,':',model3.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result4 = model4.score(x_test,y_test)
print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
r2 = r2_score(y_test,y_predict4)

print( 'r2_score4 :',r2)
print(model4,':',model4.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

# 삭제후 

# model.score: 0.9666666666666667
# r2_score1 : 0.958100558659218
# DecisionTreeClassifier() : [0.02506789 0.06761888 0.90731323]
# ===================================
# model1.score: 0.9333333333333333
# r2_score2 : 0.9162011173184358
# RandomForestClassifier() : [0.21475798 0.37988318 0.40535883]
# ===================================
# model2.score3: 0.9666666666666667
# r2_score3 : 0.958100558659218
# GradientBoostingClassifier() : [0.01294319 0.64726702 0.33978979]
# ===================================
# model4.score: 0.9666666666666667
# r2_score4 : 0.958100558659218
# XGBClassifier : [0.01042643 0.8341722  0.15540144]
# ===================================


# 삭제전 
# DecisionTreeClassifier() : [0.03338202 0.         0.56740948 0.39920851]
# RandomForestClassifier() : [0.10385929 0.03867157 0.39319982 0.46426933]
# GradientBoostingClassifier() : [0.00482361 0.01545806 0.3617882  0.61793013]
# XGBClassifier : [0.00912187 0.0219429  0.678874   0.29006115]




