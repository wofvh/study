
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.experimental import enable_halving_search_cv   # 실험버전사용할때 사용.
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingGridSearchCV

from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
n_splits =5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


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
model =HalvingGridSearchCV(RandomForestClassifier(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
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
print('acc_score:',accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 acc:',accuracy_score(y_test,y_pred_best))
print("걸린시간 :",round(end_time-start_time,4),"초")

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

# HalvingGridSearchCV
# 최적의 매개변수 : RandomForestClassifier(max_depth=10, min_samples_leaf=3)
# 최적의 파라미터: {'max_depth': 10, 'min_samples_leaf': 3}
# 최적의 점수: 0.9666666666666666
# model.score : 0.9824561403508771
# acc_score: 0.9824561403508771
# 최적의 튠 acc: 0.9824561403508771
# 걸린시간 : 38.8024 초