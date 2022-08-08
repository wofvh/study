from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import math

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

parameters = [
    {'RF__n_estimators':[100,200],'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__max_depth':[6,8,10,12],'RF__min_samples_leaf':[3,5,7]},
    {'RF__min_samples_leaf':[3,5,7],'RF__min_samples_split':[2,3,5,20]},
    {'RF__min_samples_split':[2,3,5,20]},
    {'RF__n_jobs':[-1,2,4],'RF__min_samples_leaf':[3,5,7]}
]                

from sklearn.model_selection import KFold, StratifiedKFold
n_splits=5
kFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

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

model =  RandomizedSearchCV(pipe ,parameters, cv =kFold ,verbose= 1)


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
# model.score : 0.8435754189944135
# acc_score: 0.8435754189944135
# 걸린시간 : 9.2673 초

# nopipeline 
# model.score : 0.8603351955307262
# acc_score: 0.8603351955307262
# 걸린시간 : 0.1054 초
# pipeline
# model.score : 0.8435754189944135
# acc_score: 0.8435754189944135
# 걸린시간 : 0.0985 초

# 최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=5, n_estimators=200)최적의 파라미터: {'max_depth': 6, 'min_samples_leaf': 5, 'n_estimators': 200}
# 최적의 점수: 0.8175022160937655
# model.score : 0.8379888268156425
# acc_score: 0.8379888268156425
# 최적의 튠 acc: 0.8379888268156425
# 걸린시간 : 25.6276 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestClassifier(max_depth=12, min_samples_leaf=5, n_estimators=200)
# 최적의 파라미터: {'n_estimators': 200, 'min_samples_leaf': 5, 'max_depth': 12}
# 최적의 점수: 0.8161036146951639
# model.score : 0.8379888268156425
# acc_score: 0.8379888268156425
# 최적의 튠 acc: 0.8379888268156425
# 걸린시간 : 3.9104 초