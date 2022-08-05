
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
 
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

parameters = [
    {'n_estiators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10]},
    {'min_samples_split':[2,3,5,20]},
    {'n_jobs':[-1,2,4]}
]                                                                
    

#2. 모델
# model= SVC(C=1, kernel='linear', degree=3)
model =GridSearchCV(SVC(),parameters, cv=kfold,verbose=1,       #(모델,파라미터,크로스발리데이션)
                    refit=True,n_jobs=-1)


#3. 컴파일,훈련
import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
print('최적의 매개변수 :',model.best_estimator_)
# 최적의 매개변수 : SVC(C=1, kernel='linear')
print("최적의 파라미터:",model.best_params_)
# 최적의 파라미터: {'C': 1, 'degree': 3, 'kernel': 'linear'}
print("최적의 점수:",model.best_score_)
# 최적의 파라미터: 0.9916666666666668
print('model.score :',model.score(x_test,y_test))
# model.score : 0.9666666666666667

y_predict= model.predict(x_test)
print('acc_score:',accuracy_score(y_test,y_predict))
# acc_score: 0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 acc:',accuracy_score(y_test,y_pred_best))
# 최적의 튠 acc: 0.9666666666666667

print("걸린시간 :",round(start_time-end_time,4),"초")


