
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


#2. 모델 
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRFRegressor()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
# result = model.score(x_test,y_test)
# print("model.score:",result)

from sklearn.metrics import accuracy_score, r2_score

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)

# print( 'r2_score :',r2)
# print("===================================")
print(model1,':',model1.feature_importances_)           # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 


import matplotlib.pyplot as plt 
def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align ='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Important')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    # if model == XGBRFRegressor():
    #     plt.title(model4)
    # plt.title(model)
   
model5 = 'XGBRFRegressor()'

plt.subplot(2,2,1)
plt.title(model1)
plot_feature_importances(model1)
plt.subplot(2,2,2)
plt.title(model2)
plot_feature_importances(model2)
plt.subplot(2,2,3)
plt.title(model3)
plot_feature_importances(model3)
plt.subplot(2,2,4)
plt.title(model5)
plot_feature_importances(model4)

    
plt.show()     
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