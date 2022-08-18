import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston, fetch_california_housing
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

#1. 데이터
datasets = fetch_california_housing()

# df = pd.DataFrame(datasets.data,columns=datasets.feature_names)

# print(df.head(7))

x_train, x_test,y_train,y_test = train_test_split (datasets.data,datasets.target,train_size=0.8,random_state=123,shuffle=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 .모델


lgb = LGBMRegressor()
cvb = CatBoostRegressor(verbose=0)
xg =XGBRegressor()


model = VotingRegressor(estimators=[('LGB',lgb),('CVB',cvb),('XG',xg)],)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가
y_predict = model.predict(x_test)

score = r2_score(y_test,y_predict)
print('voting결과',round(score,4))

# 결과 0.9912

classifiers = [lgb,cvb,xg]
for model2 in classifiers :
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test,y_predict)
    class_name =model2.__class__.__name__
    print('{0} 정확도:{1:.4f}'.format(class_name,score2))
    
    
# voting결과 0.853
# LGBMRegressor 정확도:0.8413
# CatBoostRegressor 정확도:0.8571
# XGBRegressor 정확도:0.8331


