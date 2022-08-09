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

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape,y.shape )

pca = PCA(n_components=20)               # 주성분 / 열축소 13 > 2개로 압축. 
x = pca.fit_transform(x)
print(x.shape,y.shape )

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)                              # 새로 생긴 피쳐에 값에 중요도를 보여줌. 
print(sum(pca_EVR))                         # 0.999998352533973 모든값을 합친 값.

cumsum = np.cumsum(pca_EVR)             
print(cumsum)
# cumsum 누적합 값이 쌓이면서 1이 된다. 
# [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791   
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835]

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
# 그래프화 


# x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123)

# #.2 모델
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from xgboost import XGBClassifier, XGBRFRegressor

# model = RandomForestRegressor()

# #3. 훈련
# model.fit(x_train,y_train) #, eval_metric='error')

# #4. 예측 
# results = model.score(x_test, y_test)
# print( " 결과", results)

# (506, 13) (506,)
#  결과 0.7684312024357363

# (506, 11) (506,)
#  결과 0.7966893548477038


