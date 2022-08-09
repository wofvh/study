# 과제
# ativation : sigmoid, relu, linear
# metrics 추가 
# earlystopping 포함.
# 성능비교
# 감상문 2줄이상 
# 구글원격 
# r2값? loss값 ? accuracy값? 
# california , diabet, boston >> 회귀모델 metrics=mse, mae 값 프린트 (relu 1.2,3 사용할 때마다 뭐가 다른지)


import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import accuracy_score
#1. 데이터

datasets = load_boston()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


#2. 모델구성

from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms:',allAlgorithms)
print('모델개수:',len(allAlgorithms))
from sklearn.metrics import r2_score
for (name,algorithm) in  allAlgorithms :
    try :
        model = algorithm()
        model.fit(x_train,y_train)
        
        y_predict = model.predict(x_test)
        
        r2 = r2_score(y_test,y_predict)
        print('r2 스코어 :', r2)
    except:
        # continue
        print(name,"은 안나온 놈")

# 모델개수: 55
# r2 스코어 : 0.7645003842515375
# r2 스코어 : 0.8326358846099354
# r2 스코어 : 0.8977171684218879
# r2 스코어 : 0.7633912164676611
# r2 스코어 : 0.7364068900582037
# r2 스코어 : 0.8442429097379944
# r2 스코어 : -0.0007435313444192904
# r2 스코어 : 0.17711506401839594
# r2 스코어 : 0.7696460772661806
# r2 스코어 : 0.7913481275532397
# r2 스코어 : 0.9459472510906538
# r2 스코어 : 0.20836937040384151
# r2 스코어 : -2.856695169473014
# r2 스코어 : 0.9105262339121646
# r2 스코어 : 0.9101594773573257
# r2 스코어 : 0.7909162259641696
# IsotonicRegression 은 안나온 놈
# r2 스코어 : 0.8146980881814008
# r2 스코어 : 0.7770067402322329
# r2 스코어 : 0.7575682701254003
# r2 스코어 : 0.7575682701254003
# r2 스코어 : 0.2553821152606973
# r2 스코어 : 0.7594055568343382
# r2 스코어 : -0.0007435313444192904
# r2 스코어 : 0.7579692443889174
# r2 스코어 : 0.7605005589228735
# r2 스코어 : 0.757969244388918
# r2 스코어 : 0.709717833839522
# r2 스코어 : 0.3860271376591802
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# r2 스코어 : 0.6577749369803669
# r2 스코어 : 0.5751842201811064
# r2 스코어 : 0.7742196789598
# r2 스코어 : -2.484314674263764
# r2 스코어 : 0.7958048688736796
# r2 스코어 : 0.7009994409696516
# r2 스코어 : 0.6742946161569346
# r2 스코어 : -0.016609681075150817
# r2 스코어 : 0.7293082320949239
# r2 스코어 : 0.4144567124477506
# r2 스코어 : 0.924828996214033
# RegressorChain 은 안나온 놈
# r2 스코어 : 0.7724176047612097
# r2 스코어 : 0.7607135401933292
# r2 스코어 : 0.8019037287995747
# r2 스코어 : 0.6839013068445225
# StackingRegressor 은 안나온 놈
# r2 스코어 : 0.7927294120868181
# r2 스코어 : 0.757969244388918
# r2 스코어 : 0.20643425204416843
# VotingRegressor 은 안나온 놈