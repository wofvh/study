from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
# lgb = LGBMRegressor()
# cvb = CatBoostRegressor(verbose=0)
# xg =XGBRegressor()

#1. 데이터 
datasets = load_boston()
x, y = datasets.data,datasets.target
print(x.shape,y.shape)


x_train, x_test, y_train, y_test = train_test_split (x,y ,train_size=0.8,random_state=1234,shuffle=True)

kfold = KFold(n_splits=5, shuffle=True,random_state=1234)

model = make_pipeline(StandardScaler(),
                      XGBRegressor()
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
                      LinearRegression()
                      )


model.fit(x_train,y_train)
print('적용후:',model.score(x_test,y_test))
scores = cross_val_score(model, x_train, y_train, cv=kfold,scoring='r2')
print('polyCV:', scores)
print('polyCVn빵:',np.mean(scores))

# 그냥: 0.7665382927362877
# CV: [0.71606004 0.67832011 0.65400513 0.56791147 0.7335664 
# ]
# CVn빵: 0.669972627809433
# (506, 104)
# 적용후: 0.925601703363665
# polyCV: [0.90649933 0.86538255 0.85156272 0.88673718 0.90526443]
# polyCVn빵: 0.8830892416737083