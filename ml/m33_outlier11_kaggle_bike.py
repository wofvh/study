# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from keras.layers.recurrent import  SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

###########이상치 처리##############
def dr_outlier(train_set):
    quartile_1 = train_set.quantile(0.25)
    quartile_3 = train_set.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
    condition = condition.any(axis=1)

    return train_set, train_set.drop(train_set.index, axis=0)

dr_outlier(train_set)
####################################


######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)# [10886 rows x 13 columns]
print(test_set)# [6493 rows x 12 columns]

##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)
y = train_set['count'] 
print(y)
print(y.shape) # (10886,)
# x = np.array(x)
# x = np.delete(x,[0,1,7], axis=1) 

# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


# print(x.shape,y.shape)
# print(datasets.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 =DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRFRegressor()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
result1 = model1.score(x_test,y_test)
print("model.score:",result1)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model1.predict(x_test)
r2 = r2_score(y_test,y_predict)

print( 'r2_score1 :',r2)
print(model1,':',model1.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result2 = model2.score(x_test,y_test)
print("model1.score:",result2)


y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test,y_predict2)

print( 'r2_score2 :',r2)
print(model2,':',model2.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result3 = model3.score(x_test,y_test)
print("model2.score3:",result3)


y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test,y_predict3)

print( 'r2_score3 :',r2)
print(model3,':',model3.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result4 = model4.score(x_test,y_test)
print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
r2 = r2_score(y_test,y_predict4)

print( 'r2_score4 :',r2)
print(model4,':',model4.feature_importances_)   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

# 삭제후 

# model.score: 0.9021949237846633
# r2_score1 : 0.9021949237846633
# DecisionTreeRegressor() : [0.04525372 0.01499567 0.10412015 0.0168487  0.02800678 0.61691358
#  0.05111164 0.03910304 0.08364672]
# ===================================
# model1.score: 0.946406555399816
# r2_score2 : 0.946406555399816
# RandomForestRegressor() : [0.04561526 0.01516876 0.0820246  0.03541214 0.03211869 0.60762559
#  0.04877114 0.04815211 0.08511172]
# ===================================
# model2.score3: 0.8692580702525858
# r2_score3 : 0.8692580702525858
# GradientBoostingRegressor() : [0.08315606 0.01061969 0.06655305 0.04688706 0.01110126 0.62440432
#  0.02416349 0.04160321 0.09151186]
# ===================================
# model4.score: 0.7221696953838107
# r2_score4 : 0.7221696953838107
# XGBRFRegressor


# 삭제전 
# model.score: 0.893539274244006
# r2_score1 : 0.893539274244006
# DecisionTreeRegressor() : [0.00423155 0.00312771 0.04284358 0.01444462 0.10139599 0.01587445
#  0.02626874 0.00915824 0.61566446 0.04872842 0.03458107 0.08368116]
# ===================================
# model1.score: 0.9458887456598607
# r2_score2 : 0.9458887456598607
# RandomForestRegressor() : [0.0058589  0.00182713 0.045541   0.0138575  0.07622974 0.04043983
#  0.02888645 0.00913296 0.61005703 0.0421705  0.04064121 0.08535774]
# ===================================
# model2.score3: 0.8682991901965896
# r2_score3 : 0.8682991901965896
# GradientBoostingRegressor() : [1.19651198e-02 3.28948974e-04 8.37764016e-02 1.06619077e-02
#  6.73458693e-02 4.58351243e-02 1.10840215e-02 8.06615251e-04
#  6.24654434e-01 2.20272528e-02 2.98930229e-02 9.16212818e-02]
# ===================================
# model4.score: 0.7220139918239472
# r2_score4 : 0.7220139918239472
# XGBRFRegressor



