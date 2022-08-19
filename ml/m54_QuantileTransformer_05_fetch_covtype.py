from sklearn.datasets import load_boston, load_breast_cancer,load_wine, fetch_covtype
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

#1. 데이터 
datasets = fetch_covtype()
x, y = datasets.data,datasets.target
print(x.shape,y.shape)


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
for sca in scaler :
    x_train = sca.fit_transform(x_train)
    x_test = sca.transform(x_test)
    model = LogisticRegression()
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test,y_predict)
    print('scaler: ', round(result,4),scaler)
    
    
exit()

# scaler:  0.9473 [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]
# scaler:  0.8946 [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]
# scaler:  0.9473 [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]
# scaler:  0.9473 [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]
# scaler:  0.9473 [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]
# scaler:  0.8946 [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]


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
