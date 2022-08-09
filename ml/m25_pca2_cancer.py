from unittest import result
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import sklearn as sk
from tqdm import trange
print(sk.__version__)
import warnings
warnings.filterwarnings(action='ignore')
# 오류가 떳을 때  ignore로 프린트 되게 하는 것. 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

model = RandomForestClassifier()

for i in range(1, 31) :                     # i = 1~30
    x = datasets.data                       
    pca = PCA(n_components=i)               # pca = n_components =30 번  반복
    x = pca.fit_transform(x)                # 
    print(x.shape)
    #print()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=666, shuffle=True
    )
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print('결과 : ', result)
    
#x = pca.fit_transform(x)
print(x.shape) # (506, 2)


# print(x.shape,y.shape )

# pca = PCA(n_components=4)               # 주성분 / 열축소 13 > 2개로 압축. 
# x = pca.fit_transform(x)
# print(x.shape,y.shape )

# # (569, 4) (569,)
# #  결과 0.9047168727029736

# # (569, 11) (569,)
# #  결과 0.8847659204811226

# # (569, 15) (569,)
# #  결과 0.8948670898763782

# # (569, 20) (569,)
# #  결과 0.8828348145673237

# # (569, 25) (569,)
# #  결과 0.8742191112596057

# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123)

# #.2 모델


# model = RandomForestRegressor()

# #3. 훈련
# model.fit(x_train,y_train) #, eval_metric='error')

# #4. 예측 
# results = model.score(x_test, y_test)
# print( " 결과", results)




