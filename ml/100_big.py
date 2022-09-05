import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = 'C:\study\_data/big/'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

train= pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

print(train,test)
print(train.isnull().sum())
print(test.isnull().sum())
x = train.drop(['Segmentation'], axis=1)
y = train['Segmentation']

print(x.shape,y.shape)



x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)

from xgboost import XGBClassifier
model = XGBClassifier(random_state=123,                 #위에있는 파라미터를 모델안에 넣을때 하는 방법
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    gamma=1)

model.fit(x_train,y_train)
from sklearn.metrics import r2_score, accuracy_score

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print('r2',r2)