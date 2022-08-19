from sklearn.datasets import load_boston, load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer,load_wine, load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
x = np.array(x)
x = np.delete(x,[4,6], axis=1)
y = np.array(y).reshape(-1, 1)

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

# 그냥: 0.8044692737430168
# CV: [0.86013986 0.79020979 0.76760563 0.75352113 0.76056338]
# CVn빵: 0.7864079582389442
# (891, 20)
# 적용후: 0.8379888268156425
# polyCV: [0.85314685 0.81818182 0.81690141 0.77464789 0.78169014]
# polyCVn빵: 0.808913621589678