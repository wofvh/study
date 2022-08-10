import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xg 
print('xgboostversion: ',xg.__version__)        # xgboostversion:  1.6.1

'''
1. iris
2. cancer
4. wine
5. fetch_covtype
6. digit
7. kaggle_titanic
'''
#1. 데이터 

datasets = fetch_covtype()
# datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape)              # (581012, 54)

# le = LabelEncoder()
# y = le.fit_transform(y)

# pca = PCA(n_components=10)       #   54 >10
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=6)
# lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x)

# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)             
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123,
                                                    stratify=y)

# lda = LinearDiscriminantAnalysis()
# lda.fit(x_train,y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)

print(np.unique(y_train, return_counts=True))               # array([1, 2, 3, 4, 5, 6, 7] > 
                                                            # array([0, 1, 2, 3, 4, 5, 6]
#2. 모델
from xgboost import XGBClassifier ,XGBRFRegressor
model = XGBRFRegressor(tree_method='gpu_hist',
                      predictor='gpu_predictor',
                      gpu_id=0)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 평가 예측

results= model.score(x_test,y_test)
print("결과 :",results)
print("시간 :", end-start )

# LinearDiscriminantAnalysis()
# 결과 : 0.2920645715998115
# 시간 : 34.98989510536194

# pca = PCA(n_components=10)       
# 결과 : 0.4617027185456226
# 시간 : 34.25459003448486


