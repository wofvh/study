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
from sklearn.metrics import r2_score
#1. 데이터

datasets = load_boston()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)
parameters = [
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__min_samples_leaf':[3,5,7],'RF__min_samples_split':[2,3,5,20]},
    {'RF__min_samples_split':[2,3,5,20]},
    {'RF__n_jobs':[-1,2,4],'RF__min_samples_leaf':[3,5,7]}
]                

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kFold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# model = RandomForestClassifier()
# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.
pipe = Pipeline([('minmax',MinMaxScaler()),('RF',RandomForestRegressor())],verbose=1)


#3. 컴파일 훈련
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

model =  RandomizedSearchCV(pipe ,parameters, cv =kFold ,verbose= 1)


import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
#4. 평가 예측
result = model.score(x_test, y_test)

print('model.score :',model.score(x_test,y_test))

y_predict= model.predict(x_test)
print('r2_score:',r2_score(y_test,y_predict))

print("걸린시간 :",round(end_time-start_time,4),"초")

# RandomizedSearchCV + KFold + pipe
# model.score : 0.9131601653020569
# r2_score: 0.9131601653020569
# 걸린시간 : 7.7635 초


# nopipeline 
# model.score : 0.9252623849629499
# r2_score: 0.9252623849629499
# 걸린시간 : 0.1824 초
# pipeline
# model.score : 0.9221627285564329
# r2_score: 0.9221627285564329
# 걸린시간 : 0.1736 초


# 최적의 매개변수 : RandomForestRegressor()
# 최적의 파라미터: {'min_samples_split': 2}
# 최적의 점수: 0.8490237656609413
# model.score : 0.9342979049991655
# r2_score: 0.9342979049991655
# 최적의 튠 acc: 0.9342979049991655
# 걸린시간 : 19.1774 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestRegressor(min_samples_split=5)
# 최적의 파라미터: {'min_samples_split': 5}
# 최적의 점수: 0.8390032209315368
# model.score : 0.9227030971363911
# r2_score: 0.9227030971363911
# 최적의 튠 acc: 0.9227030971363911
# 걸린시간 : 3.5692 초