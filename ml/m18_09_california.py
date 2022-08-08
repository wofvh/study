
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
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성 
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
# model.score : 0.804706749076605
# r2_score: 0.804706749076605
# 걸린시간 : 205.5625 초

# nopipeline 
# model.score : 0.8049002018837632
# r2_score: 0.8049002018837632
# 걸린시간 : 6.7436 초
# pipeline
# model.score : 0.8058216690517961
# r2_score: 0.8058216690517961
# 걸린시간 : 6.7416 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestRegressor(min_samples_leaf=5, min_samples_split=5)
# 최적의 파라미터: {'min_samples_split': 5, 'min_samples_leaf': 5}
# 최적의 점수: 0.7995187294935869
# model.score : 0.800858810087763
# r2_score: 0.800858810087763
# 최적의 튠 acc: 0.800858810087763
# 걸린시간 : 42.9752 초
