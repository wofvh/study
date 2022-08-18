from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression     
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 
                        index_col=0)                       

test_set = pd.read_csv(path + 'test.csv',                                   
                       index_col=0)

train_set = train_set.fillna(train_set.mean())       
test_set = test_set.fillna(test_set.mean())
x = train_set.drop(['count'], axis=1)                    
y = train_set['count'] 
x = np.array(x)
x = np.delete(x,[2,3,4], axis=1)  

from sklearn.model_selection import train_test_split, KFold , StratifiedKFold


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


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
# score1 : 0.7865685604346746
# BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score2 : 0.7790456939802397
# BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score3 : 0.7377293548880837
# BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# acc : 0.7443257132757819
# BaggingRegressor(base_estimator=XGBRFRegressor
                 
# 삭제후 

# model.score: 0.6300146059727504
# r2_score1 : 0.6300146059727504
# DecisionTreeRegressor() : [0.58889837 0.19193815 0.07812078 0.0589318  0.04272128 0.03938963]
# ===================================
# model1.score: 0.7821477073674483
# r2_score2 : 0.7821477073674483
# RandomForestRegressor() : [0.59613307 0.19878038 0.06884772 0.05845012 0.04331326 0.03447545]
# ===================================
# model2.score3: 0.7953047052028843
# r2_score3 : 0.7953047052028843   
# RandomForestRegressor() : [0.59692309 0.19396559 0.06771564 0.05766864 0.04795155 0.03577549]
# ===================================
# model4.score: 0.7438176570900675
# r2_score4 : 0.7438176570900675
# XGBRFRegressor


# 삭제전 
# model.score: 0.6203555250587188
# r2_score1 : 0.6203555250587188
# DecisionTreeRegressor() : [0.58266757 0.17451747 0.02718274 0.01550531 0.02803954 0.04941452  
#  0.04749699 0.04161734 0.03355851]
# ===================================
# model1.score: 0.7961345753061605
# r2_score2 : 0.7961345753061605
# RandomForestRegressor() : [0.58372849 0.18168007 0.02193944 0.03235003 0.03393578 0.04210459  
#  0.04164655 0.03494641 0.02766864]
# ===================================
# model2.score3: 0.7954735396341172
# r2_score3 : 0.7954735396341172
# RandomForestRegressor() : [0.57913503 0.19415576 0.0211313  0.03100085 0.03570498 0.04025897  
#  0.03780124 0.0357234  0.02508846]
# ===================================
# model4.score: 0.7576400722892229
# r2_score4 : 0.7576400722892229
# XGBRFRegressor




