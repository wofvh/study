from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 # pandas : 엑셀땡겨올때 씀 python 지원하는 엑셀을 불러오는 기능.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_halving_search_cv   # 실험버전사용할때 사용.
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)

train_set = train_set.fillna(train_set.mean())       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값

x = train_set.drop(['count'], axis=1)                    # drop 데이터에서 ''사이 값 빼기

y = train_set['count'] 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.05,                                
    shuffle=True, random_state =58525)


from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
n_splits =5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

parameters = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7]},
    {'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7]},
    {'min_samples_leaf':[3,5,7],'min_samples_split':[2,3,5,20]},
    {'min_samples_split':[2,3,5,20]},
    {'n_jobs':[-1,2,4],'min_samples_leaf':[3,5,7]}
]                                                   
    
from sklearn.model_selection import RandomizedSearchCV
#2. 모델
# model= SVC(C=1, kernel='linear', degree=3)
model =HalvingRandomSearchCV(RandomForestRegressor(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
                    refit=True,n_jobs=-1)


#3. 컴파일,훈련
import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
print('최적의 매개변수 :',model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("최적의 점수:",model.best_score_)
print('model.score :',model.score(x_test,y_test))

y_predict= model.predict(x_test)
print('r2_score:',r2_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 acc:',r2_score(y_test,y_pred_best))
print("걸린시간 :",round(end_time-start_time,4),"초")

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터: {'min_samples_leaf': 3, 'max_depth': 12}
# 최적의 점수: 0.7736872015832216
# model.score : 0.7447009390145163
# r2_score: 0.7447009390145163
# 최적의 튠 acc: 0.7447009390145163
# 걸린시간 : 4.5875 초

# HalvingGridSearchCV
# 최적의 매개변수 : RandomForestRegressor(min_samples_split=3)
# 최적의 파라미터: {'min_samples_split': 3}
# 최적의 점수: 0.7790368697472585
# model.score : 0.7493383254340198
# r2_score: 0.7493383254340198
# 최적의 튠 acc: 0.7493383254340198
# 걸린시간 : 44.9165 초

# HalvingRandomSearchCV
# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=5, n_estimators=200)
# 최적의 파라미터: {'n_estimators': 200, 'min_samples_leaf': 5, 'max_depth': 8}
# 최적의 점수: 0.6927120107180137
# model.score : 0.7441347977548585
# r2_score: 0.7441347977548585
# 최적의 튠 acc: 0.7441347977548585
# 걸린시간 : 80.2992 초

