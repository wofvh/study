from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.pipeline import make_pipeline

#1. 데이터 
datasets = load_iris()
x, y = datasets.data,datasets.target
print(x.shape,y.shape)


x_train, x_test, y_train, y_test = train_test_split (x,y ,train_size=0.8,random_state=123,shuffle=True)

kfold = KFold(n_splits=5, shuffle=True,random_state=123)

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

kfold = KFold(n_splits=5, shuffle=True, random_state=123)

model = make_pipeline(StandardScaler(),
                      LogisticRegression()
                      )


model.fit(x_train,y_train)
print('적용후:',model.score(x_test,y_test))
scores = cross_val_score(model, x_train, y_train, cv=kfold,scoring='accuracy')
print('polyCV:', scores)
print('polyCVn빵:',np.mean(scores))

# 그냥: 0.938638309162935
# CV: [nan nan nan nan nan]
# CVn빵: nan
# (150, 14)
# 적용후: 0.9666666666666667
# polyCV: [1.         1.         0.95833333 1.         0.95833333]
# polyCVn빵: 0.9833333333333334