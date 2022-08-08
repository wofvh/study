import numpy as np
from sklearn.datasets import load_iris


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold , StratifiedKFold

x_train,x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=1234)

parameters = [
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__min_samples_leaf':[3,5,7],'RF__min_samples_split':[2,3,5,20]},
    {'RF__min_samples_split':[2,3,5,20]},
    {'RF__n_jobs':[-1,2,4],'RF__min_samples_leaf':[3,5,7]}
]                

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestClassifier())],verbose=1)


#3. 컴파일 훈련
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

model = RandomizedSearchCV(pipe ,parameters, cv =5 ,verbose= 1)


import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
#4. 평가 예측
result = model.score(x_test, y_test)

print('model.score :',model.score(x_test,y_test))

from sklearn.metrics import accuracy_score
y_predict= model.predict(x_test)
print('acc_score:',accuracy_score(y_test,y_predict))

print("걸린시간 :",round(end_time-start_time,4),"초")

# RandomizedSearchCV + KFold + pipe
# model.score : 1.0
# acc_score: 1.0
# 걸린시간 : 9.346 초

# nopipeline 
# model.score: 1.0
# pipeline
# model.score: 1.0


# 최적의 점수: 0.9583333333333334
# model.score : 0.9333333333333333
# acc_score: 0.9333333333333333
# 최적의 튠 acc: 0.9333333333333333
# 걸린시간 : 14.5051 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestClassifier(min_samples_leaf=5, min_samples_split=20)
# 최적의 파라미터: {'min_samples_split': 20, 'min_samples_leaf': 5}
# 최적의 점수: 0.95
# model.score : 0.9333333333333333
# acc_score: 0.9333333333333333
# 최적의 튠 acc: 0.9333333333333333
# 걸린시간 : 2.873 초

# HalvingGridSearchCV
# 최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=3)
# 최적의 파라미터: {'max_depth': 6, 'min_samples_leaf': 3}
# 최적의 점수: 0.9666666666666668
# model.score : 0.9666666666666667
# acc_score: 0.9666666666666667
# 최적의 튠 acc: 0.9666666666666667
# 걸린시간 : 34.9192 초



