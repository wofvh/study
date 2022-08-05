
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.experimental import enable_halving_search_cv   # 실험버전사용할때 사용.
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
n_splits =5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.model_selection import RandomizedSearchCV
parameters = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7]},
    {'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7]},
    {'min_samples_leaf':[3,5,7],'min_samples_split':[2,3,5,20]},
    {'min_samples_split':[2,3,5,20]},
    {'n_jobs':[-1,2,4],'min_samples_leaf':[3,5,7]}
]                                                   
    

#2. 모델
# model= SVC(C=1, kernel='linear', degree=3)
model =HalvingRandomSearchCV(RandomForestClassifier(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
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

# HalvingRandomSearchCV
# 최적의 매개변수 : RandomForestClassifier(max_depth=12, min_samples_leaf=5)
# 최적의 파라미터: {'min_samples_leaf': 5, 'max_depth': 12}
# 최적의 점수: 0.9444444444444444
# model.score : 0.9666666666666667
# acc_score: 0.9666666666666667
# 최적의 튠 acc: 0.9666666666666667
# 걸린시간 : 8.0606 초

