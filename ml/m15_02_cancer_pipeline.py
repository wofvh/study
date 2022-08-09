
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import make_pipeline

# model = RandomForestClassifier()

model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.


#3. 컴파일,훈련
model.fit(x_train,y_train)

import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()

print('model.score :',model.score(x_test,y_test))

y_predict= model.predict(x_test)
print('acc_score:',accuracy_score(y_test,y_predict))

print("걸린시간 :",round(end_time-start_time,4),"초")

# nopipeline 
# model.score : 0.9824561403508771
# acc_score: 0.9824561403508771
# 걸린시간 : 0.1198 초
# pipeline
# model.score : 0.9736842105263158
# acc_score: 0.9736842105263158
# 걸린시간 : 0.1282 초


# 최적의 매개변수 : RandomForestClassifier(max_depth=10, min_samples_leaf=3)
# 최적의 파라미터: {'max_depth': 10, 'min_samples_leaf': 3, 'n_estimators': 100}
# 최적의 점수: 0.9582417582417584
# model.score : 0.9824561403508771
# acc_score: 0.9824561403508771
# 최적의 튠 acc: 0.9824561403508771
# 걸린시간 : 21.5003 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestClassifier(max_depth=6, min_samples_leaf=5, n_estimators=200)최적의 파라미터: {'n_estimators': 200, 'min_samples_leaf': 5, 'max_depth': 6}
# 최적의 점수: 0.956043956043956
# model.score : 0.9736842105263158
# acc_score: 0.9736842105263158
# 최적의 튠 acc: 0.9736842105263158
# 걸린시간 : 3.3383 초