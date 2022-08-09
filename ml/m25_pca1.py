from unittest import result
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import sklearn as sk
from tqdm import trange
print(sk.__version__)
import warnings
warnings.filterwarnings(action='ignore')
# 오류가 떳을 때  ignore로 프린트 되게 하는 것. 

#1. 데이터 
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape,y.shape )

pca = PCA(n_components=11)               # 주성분 / 열축소 13 > 2개로 압축. 
x = pca.fit_transform(x)
print(x.shape,y.shape )

# (506, 13) (506,)
#  결과 0.7684312024357363

# (506, 11) (506,)
#  결과 0.7966893548477038


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123)

#.2 모델
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor

model = RandomForestRegressor()

#3. 훈련
model.fit(x_train,y_train) #, eval_metric='error')

#4. 예측 
results = model.score(x_test, y_test)
print( " 결과", results)




