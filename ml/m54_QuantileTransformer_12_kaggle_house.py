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
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
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

# scaler: StandardScaler() model: <catboost.core.CatBoostRegressor object at 0x000001F994350B50> result: 0.8998
# scaler: StandardScaler() model: RandomForestRegressor() result: 0.8688
# scaler: StandardScaler() model: LinearRegression() result: 0.5226
# scaler: MinMaxScaler()   model: <catboost.core.CatBoostRegressor object at 0x000001F994350B50> result: 0.8995
# scaler: MinMaxScaler()   model: RandomForestRegressor() result: 0.8655
# scaler: MinMaxScaler()   model: LinearRegression() result: 0.5226
# scaler: RobustScaler()   model: <catboost.core.CatBoostRegressor object at 0x000001F994350B50> result: 0.8995
# scaler: RobustScaler()   model: RandomForestRegressor() result: 0.8663
# scaler: RobustScaler()   model: LinearRegression() result: 0.5226
# scaler: MaxAbsScaler()   model: <catboost.core.CatBoostRegressor object at 0x000001F994350B50> result: 0.8995
# scaler: MaxAbsScaler()   model: RandomForestRegressor() result: 0.8696
# scaler: MaxAbsScaler()   model: LinearRegression() result: 0.5226
# scaler: QuantileTransformer() model: <catboost.core.CatBoostRegressor object at 0x000001F994350B50> result: 0.902
# scaler: QuantileTransformer() model: RandomForestRegressor() result: 0.8686
# scaler: QuantileTransformer() model: LinearRegression() result: 0.8247
# scaler: PowerTransformer() model: <catboost.core.CatBoostRegressor object at 0x000001F994350B50> result: 0.9005
# scaler: PowerTransformer() model: RandomForestRegressor() result: 0.8638
# scaler: PowerTransformer() model: LinearRegression() result: 0.8199


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
