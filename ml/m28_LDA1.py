import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer,fetch_covtype
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#1. 데이터 

datasets = load_boston()

x = datasets.data
y = datasets.target
print(x.shape)              # (581012, 54)

le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=20)       #   54 >10
# x = pca.fit_transform(x)

# lda = LinearDiscriminantAnalysis()


# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)             
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=123,
                                                    stratify=y)

lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)



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


# XGBClassifier
# 결과: 0.8695988915948814
# 시간 : 6.970503330230713

# XGBClassifier
# pca = PCA(n_components=10)
# 결과: 0.8406065247885166
# 시간 : 4.496622323989868

# XGBClassifier
# pca = PCA(n_components=20)   
# 결과: 0.8855279123602661
# 시간 : 5.378031492233276