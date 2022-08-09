
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 # pandas : 엑셀땡겨올때 씀 python 지원하는 엑셀을 불러오는 기능.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
# model.score : 0.7492474127761952
# r2_score: 0.7492474127761952
# 걸린시간 : 0.3168 초

# pipeline
# model.score : 0.7523934687251793
# r2_score: 0.7523934687251793
# 걸린시간 : 0.3393 초

# RandomizedSearchCV
# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터: {'min_samples_leaf': 3, 'max_depth': 12}
# 최적의 점수: 0.7736872015832216
# model.score : 0.7447009390145163
# r2_score: 0.7447009390145163
# 최적의 튠 acc: 0.7447009390145163
# 걸린시간 : 4.5875 초


