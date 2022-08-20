from tabnanny import verbose
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes,fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler ,RobustScaler, QuantileTransformer , PowerTransformer,MaxAbsScaler
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score,r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
# XGBRegressor,LGBMRegressor,CatBoostRegressor,RandomForestRegressor
#1. 데이터 
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

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
x_train, x_test, y_train, y_test = train_test_split (x,y ,train_size=0.8,
                                                     random_state=1234,
                                                     shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = QuantileTransformer()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

scaler = PowerTransformer(method='yeo-johnson')     #디폴트
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# scaler = PowerTransformer(method='box-cox')
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler ,RobustScaler, QuantileTransformer , PowerTransformer,MaxAbsScaler
scaler = [StandardScaler(),MinMaxScaler(),RobustScaler(),MaxAbsScaler(),QuantileTransformer(),PowerTransformer(method='yeo-johnson')]
models = [CatBoostRegressor(verbose=0),RandomForestRegressor(verbose=0),LinearRegression()]
for sca in scaler :
    x_train = sca.fit_transform(x_train)
    x_test = sca.transform(x_test)
    for mod in models:
        model = mod
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        result = r2_score(y_test,y_predict)
        print('scaler:',sca,'model:',mod,'result:', round(result,4))
    
    
exit()

# scaler: StandardScaler() model: <catboost.core.CatBoostRegressor object at 0x000001F225E49A00> result: 0.959
# scaler: StandardScaler() model: RandomForestRegressor() result: 0.9543
# scaler: StandardScaler() model: LinearRegression() result: 0.3903
# scaler: MinMaxScaler() model: <catboost.core.CatBoostRegressor object at 0x000001F225E49A00> result: 0.959
# scaler: MinMaxScaler() model: RandomForestRegressor() result: 0.9537
# scaler: MinMaxScaler() model: LinearRegression() result: 0.3903
# scaler: RobustScaler() model: <catboost.core.CatBoostRegressor object at 0x000001F225E49A00> result: 0.959
# scaler: RobustScaler() model: RandomForestRegressor() result: 0.9537
# scaler: RobustScaler() model: LinearRegression() result: 0.3903
# scaler: MaxAbsScaler() model: <catboost.core.CatBoostRegressor object at 0x000001F225E49A00> result: 0.959
# scaler: MaxAbsScaler() model: RandomForestRegressor() result: 0.9538
# scaler: MaxAbsScaler() model: LinearRegression() result: 0.3903
# scaler: QuantileTransformer() model: <catboost.core.CatBoostRegressor object at 0x000001F225E49A00> result: 0.959
# scaler: QuantileTransformer() model: RandomForestRegressor() result: 0.9537
# scaler: QuantileTransformer() model: LinearRegression() result: 0.396
# scaler: PowerTransformer() model: <catboost.core.CatBoostRegressor object at 0x000001F225E49A00> result: 0.959
# scaler: PowerTransformer() model: RandomForestRegressor() result: 0.954
# scaler: PowerTransformer() model: LinearRegression() result: 0.4003


#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#.3 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
y_predict = model.predict(x_test)
result = r2_score(y_test,y_predict)
print('그냥: ', round(result,4))
# 그냥:  0.7665382927362877

#################log 변환 ################################
exit()
df = pd.DataFrame(datasets.data,columns=[datasets.feature_names])
print(df)   # [506 rows x 13 columns]

import matplotlib.pyplot as plt

df.plot.box()
plt.title('boston')
plt.xlabel('Feature')
plt.ylabel('data')
plt.show()

# print(df['B'].head())
df['B'] = np.log1p(df['B'])
# print(df['B'].head())
                                            # 그냥:  0.7665382927362877
# df['CRIM'] = np.log1p(df['CRIM'])         # log변환:  0.759582163653457
df['ZN'] = np.log1p(df['ZN'])               # log변환:  0.7733890810577744
df['TAX'] = np.log1p(df['TAX'])             # log변환:  0.7669259619292876
                                            # (B,ZN,TAX) log변환:   0.7785



x_train, x_test, y_train, y_test = train_test_split (df,y ,train_size=0.8,
                                                     random_state=1234,
                                                     shuffle=True)






#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#.3 훈련 
model.fit(x_train,y_train)

#4. 평가 예측
y_predict = model.predict(x_test)
result = r2_score(y_test,y_predict)
print('log변환: ',round(result,4))
exit()
# model = LinearRegression()
# 그냥:  0.7665382927362877
# log변환:  0.7710827448613001

# model = RandomForestRegressor()
# 그냥:  0.9180727695470997
# log변환:  0.9142464936856531
