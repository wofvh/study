# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)



# x = np.array(x)
# y = np.array(y) 

# x = np.delete(x,[10,11,12,13,14,15,16,17,18,19,20,21,23,29], axis=1) 
# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


print(x.shape,y.shape)
print(datasets.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 
from sklearn.ensemble import BaggingClassifier ,BaggingRegressor # 한가지 모델을 여러번 돌리는 것(파라미터 조절).,
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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
# BaggingRegressor
# score1 : 0.5271810557713784
# BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score2 : 0.5542218926377125
# BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score3 : 0.445266620624321
# BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# acc : 0.5567726467743137
# BaggingRegressor(base_estimator=XGBRFRegressor

# 삭제후 
# model.score: 0.9649122807017544
# accuracy_score : 0.9649122807017544
# DecisionTreeClassifier() 
# ===================================
# model2.score: 0.9736842105263158
# accuracy2_score : 0.9736842105263158
# RandomForestClassifier() 
# ===================================
# model3.score: 0.9736842105263158
# accuracy3_score : 0.9736842105263158
# GradientBoostingClassifier() 
# ===================================
# model4.score: 0.9736842105263158
# accuracy4_score : 0.9736842105263158
# XGBClassifier 

# 삭제전 
# model.score: 0.956140350877193
# accuracy_score : 0.956140350877193
# DecisionTreeClassifier() :
# ===================================
# model2.score: 0.9912280701754386
# accuracy2_score : 0.9912280701754386
# RandomForestClassifier() :
# ===================================
# model3.score: 0.9736842105263158
# accuracy3_score : 0.9736842105263158
# GradientBoostingClassifier() :
# ===================================
# model4.score: 0.9649122807017544
# accuracy4_score : 0.9649122807017544
# XGBClassifier




