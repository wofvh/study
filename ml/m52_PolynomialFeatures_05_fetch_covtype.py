from sklearn.datasets import load_boston, load_iris, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

#1. 데이터 
datasets = fetch_covtype()
x, y = datasets.data,datasets.target
print(x.shape,y.shape)


x_train, x_test, y_train, y_test = train_test_split (x,y ,train_size=0.8,random_state=123,shuffle=True)

kfold = KFold(n_splits=5, shuffle=True,random_state=1234)

model = make_pipeline(StandardScaler(),
                      LogisticRegression()
                      )

from sklearn.model_selection import cross_val_score
model.fit(x_train,y_train)
print('그냥:',model.score(x_test,y_test))
scores = cross_val_score(model, x_train, y_train, cv=kfold,scoring='accuracy')
print('CV:', scores)
print('CVn빵:',np.mean(scores))


#2. 모델


#######################################PolymialFeatures 후 ############################################


pf = PolynomialFeatures(degree=2,include_bias=False)
xp = pf.fit_transform(x)
print(xp.shape)


x_train, x_test, y_train, y_test = train_test_split (xp,y ,train_size=0.8,random_state=123,shuffle=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

model = make_pipeline(StandardScaler(),
                      LogisticRegression()
                      )


model.fit(x_train,y_train)
print('적용후:',model.score(x_test,y_test))
scores = cross_val_score(model, x_train, y_train, cv=kfold,scoring='accuracy')
print('polyCV:', scores)
print('polyCVn빵:',np.mean(scores))

# 그냥: 0.7244993674862095
# CV: [0.72403778 0.72455412 0.72472623 0.72440352 0.72157141]
# CVn빵: 0.7238586122747181
# (581012, 1539)
# 적용후: 0.7631644621911655
# polyCV: [0.76219315 0.76246208 0.76522665 0.76318281 0.76190015]
# polyCVn빵: 0.7629929689553832