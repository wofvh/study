from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

# matplotlib.rcParams['font.family']='Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus']=False
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

x_train, x_test, y_train, y_test = train_test_split (x,y ,train_size=0.8,random_state=1234,shuffle=True)

kfold = KFold(n_splits=5, shuffle=True,random_state=1234)

model = make_pipeline(StandardScaler(),
                      RandomForestRegressor()
                      )

from sklearn.model_selection import cross_val_score
model.fit(x_train,y_train)
print('그냥:',model.score(x_test,y_test))
scores = cross_val_score(model, x_train, y_train, cv=kfold,scoring='r2')
print('CV:', scores)
print('CVn빵:',np.mean(scores))

# 그냥: 0.7665382927362877
# CV: [0.71606004 0.67832011 0.65400513 0.56791147 0.7335664 
# ]
# CVn빵: 0.669972627809433

#2. 모델


#######################################PolymialFeatures 후 ############################################


pf = PolynomialFeatures(degree=2,include_bias=False)
xp = pf.fit_transform(x)
print(xp.shape)


x_train, x_test, y_train, y_test = train_test_split (xp,y ,train_size=0.8,random_state=1234,shuffle=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

model = make_pipeline(StandardScaler(),
                      RandomForestRegressor()
                      )


model.fit(x_train,y_train)
print('적용후:',model.score(x_test,y_test))
scores = cross_val_score(model, x_train, y_train, cv=kfold,scoring='r2')
print('polyCV:', scores)
print('polyCVn빵:',np.mean(scores))

# 그냥: 0.3907411587845723
# CV: [0.38773729 0.39666672 0.39269986 0.39072971 0.36547995]
# CVn빵: 0.3866627065407851
# (10886, 90)
# 적용후: 0.5662966427660571
# polyCV: [0.5533226  0.55278519 0.54452828 0.5433036  0.51057148]
# polyCVn빵: 0.5409022280502567