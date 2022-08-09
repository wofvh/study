import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
import pandas as pd
#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


x= pd.DataFrame(x,columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRFRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 예측
result = model.score(x_test,y_test)
print("model.score:",result)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print( 'r2_score :',r2)
print("===================================")
print(model,':',model.feature_importances_)           # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 


import matplotlib.pyplot as plt 
# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features),model.feature_importances_, align ='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel('Feature Important')
#     plt.ylabel('Features')
#     plt.ylim(-1,n_features)
#     plt.title(model)
    
# plot_feature_importances(model)
# plt.show()     


from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()
# DecisionTreeClassifier() : [0.03338202 0.         0.56740948 0.39920851]
# RandomForestClassifier() : [0.10385929 0.03867157 0.39319982 0.46426933]
# GradientBoostingClassifier() : [0.00482361 0.01545806 0.3617882  0.61793013]
# XGBClassifier : [0.00912187 0.0219429  0.678874   0.29006115]

# DecisionTreeRegressor() : [0.07678498 0.01395182 0.34760333 0.06798322 0.03774824 0.092667420.06552908 0.01802744 0.16105957 0.11864489]
# model.score: -0.13877944766840034
# r2_score : -0.13877944766840034

# RandomForestRegressor() : [0.0552437  0.01228313 0.34441867 0.07955716 0.0496346  0.05854343 0.06350405 0.03253382 0.21798229 0.08629914]
# model.score: 0.4436332910867421
# r2_score : 0.4436332910867421

# GradientBoostingRegressor() : [0.04992045 0.01077472 0.30284172 0.11208124 0.0281211  0.0556483 0.04015688 0.01819239 0.33792526 0.04433795]
# model.score: 0.5543923878970336
# r2_score : 0.5543923878970336

# XGBRFRegressor : [0.02540992 0.02556245 0.20697184 0.07726039 0.04551006 0.06179914 0.06249548 0.09776632 0.29795545 0.09926897]
# model.score: 0.5492333048972089
# r2_score : 0.5492333048972089
