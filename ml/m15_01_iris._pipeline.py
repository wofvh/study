import numpy as np
from sklearn.datasets import load_iris


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target


x_train,x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=1234)

from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline

model = RandomForestClassifier()

# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.

#3. 컴파일 훈련

model.fit(x_train,y_train)

#4. 평가 예측
result = model.score(x_test, y_test)

print('model.score:',result )

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




