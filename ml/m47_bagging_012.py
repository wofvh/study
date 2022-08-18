# 결과비교 
# DecisionTree
# 기존 acc : 
# 컬럼삭제후 acc : 
# 4개 모델 비교 


from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 대문자 class  암시가능.
from sklearn.preprocessing import MaxAbsScaler, RobustScaler  
import matplotlib
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
path = './_data/house/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_set.drop(drop_cols, axis = 1, inplace =True)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
print(train_set)

print(train_set.shape) #(1459, 10)

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
train_set = train_set.fillna(train_set.median())
print(train_set.isnull().sum())
print(train_set.shape)
test_set = test_set.fillna(test_set.median())

x = train_set.drop(['SalePrice'],axis=1) #axis는 컬럼 
print(x.columns)
print(x.shape) #(1460, 75)

y = train_set['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.89, shuffle = True, random_state = 68
 )
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
x = np.array(x)
x = np.delete(x,[1,3,5,7,8,9,11,15,18], axis=1) 
# train_set.drop(drop_cols, axis = 1, inplace =True)

# y = np.array(y) 

# x = np.delete(x,1, axis=1) 
# x = np.delete(x,4, axis=1) 

# y = np.delete(y,1, axis=1) 


# print(x.shape,y.shape)
# print(datasets.feature_names)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 
from sklearn.ensemble import BaggingClassifier ,BaggingRegressor # 한가지 모델을 여러번 돌리는 것(파라미터 조절).,
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

model1 = BaggingRegressor(DecisionTreeRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model2 = BaggingRegressor(RandomForestRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model3 = BaggingRegressor(KNeighborsRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )

model4 = BaggingRegressor(XGBRFRegressor(),
                          n_estimators=100, 
                          n_jobs=1,
                          random_state=123
                          )


# model1 = DecisionTreeClassifier()
# model2 = RandomForestClassifier()
# model3 = GradientBoostingClassifier()
# model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측
result1 = model1.score(x_test,y_test)
# print("model1.score:",result1)

from sklearn.metrics import accuracy_score, r2_score

y_predict = model1.predict(x_test)
score1 = r2_score(y_test,y_predict)

print( 'score1 :',score1)
print(model1) 
print("===================================")

result2 = model2.score(x_test,y_test)
# print("model2.score:",result2)


y_predict2 = model2.predict(x_test)
score2 = r2_score(y_test,y_predict2)

print( 'score2 :',score2)
print(model2) 
print("===================================")

result3 = model3.score(x_test,y_test)
# print("model3.score3:",result3)


y_predict3 = model3.predict(x_test)
score3 = r2_score(y_test,y_predict3)

print( 'score3 :',score3)
print(model3)
print("===================================")

result4 = model4.score(x_test,y_test)
# print("model4.score:",result4)


y_predict4 = model4.predict(x_test)
score4 = r2_score(y_test,y_predict4)

print( 'acc :',score4)
print(model4) 
print("===================================")
# BaggingRegressor
# score1 : 0.8500008730820914
# BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score2 : 0.8445862836276797
# BaggingRegressor(base_estimator=RandomForestRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# score3 : 0.768268768409837
# BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=100,
#                  n_jobs=1, random_state=123)
# ===================================
# acc : 0.8321068609651805
# BaggingRegressor(base_estimator=XGBRFRegressor



# 삭제후 

# model.score: 0.7169631375763239
# r2_score1 : 0.7169631375763239
# DecisionTreeRegressor() :
# ===================================
# model1.score: 0.9021085021563747
# r2_score2 : 0.9021085021563747
# RandomForestRegressor() :
# ===================================
# model2.score3: 0.9099405479415681
# r2_score3 : 0.9099405479415681
# GradientBoostingRegressor() :
# ===================================
# model4.score: 0.8735940936527683
# r2_score4 : 0.8735940936527683
# XGBRFRegressor

# 삭제전 
# model.score: 0.6989610731651827
# r2_score1 : 0.6989610731651827
# DecisionTreeRegressor() :
# ===================================
# model1.score: 0.8934932671070577
# r2_score2 : 0.8934932671070577
# RandomForestRegressor() :
# ===================================
# model2.score3: 0.9098570173287146
# r2_score3 : 0.9098570173287146
# GradientBoostingRegressor() :
# ===================================
# model4.score: 0.8735940936527683
# r2_score4 : 0.8735940936527683
# XGBRFRegressor




