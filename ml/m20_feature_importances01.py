import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=1234,shuffle=True)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier           # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model1= GradientBoostingClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 예측
result = model.score(x_test,y_test)
print("model.score:",result)

from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)

print( 'accuracy_score :',acc)
print("===================================")
print(model,':',model.feature_importances_)           
# 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print(model)

# DecisionTreeClassifier() : [0.03338202 0.         0.56740948 0.39920851]
# RandomForestClassifier() : [0.10385929 0.03867157 0.39319982 0.46426933]
# GradientBoostingClassifier() : [0.00482361 0.01545806 0.3617882  0.61793013]
# XGBClassifier : [0.00912187 0.0219429  0.678874   0.29006115]