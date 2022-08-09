
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
datasets = fetch_covtype()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import make_pipeline

model = RandomForestClassifier()

# model = make_pipeline(MinMaxScaler(),RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.


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
# model.score : 0.9552421193945079
# acc_score: 0.9552421193945079
# 걸린시간 : 115.4285 초
# pipeline
# model.score : 0.9566878652014148
# acc_score: 0.9566878652014148
# 걸린시간 : 116.1415 초

# 최적의 매개변수 : RandomForestClassifier()
# 최적의 파라미터: {'min_samples_split': 2}
# 최적의 점수: 0.9508357197422288
# model.score : 0.9557670628124919
# acc_score: 0.9557670628124919
# 최적의 튠 acc: 0.9557670628124919
# 걸린시간 : 7088.3054 초