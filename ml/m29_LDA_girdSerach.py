import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost
from sklearn.experimental import enable_halving_search_cv   # 실험버전사용할때 사용. 
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PCA
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻

x = x.reshape(70000, 28*28)
pca = PCA(n_components=154)  # 칼럼이 28*28개의 벡터로 압축이됨
x = pca.fit_transform(x)

x_train = x[:60000]
x_test = x[60000:]
print(x_train.shape)

# (x_train,y_train),(x_test,y_test) =mnist.load_data()       # _를 사용하면 사용하지않겠다. 
# print(x_train.shape,x_test.shape)   # (60000, 28, 28) (10000, 28, 28)
 
# x_train = x_train.reshape(60000,784)
# x_test = x_test.reshape(10000,784)

# pca = PCA(n_components=50)   
# x_train= pca.fit_transform(x_train)
# x_test= pca.transform(x_test) 

lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x)

# print(x_train.shape,x_test.shape)       # (60000, 331) (10000, 331)
# print(np.unique(y_train, return_counts=True)) 
n_splits =5 
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [ 
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.3,0.001,0.01],
     "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1,0.001,0.01],
     "max_depth":[4,5,6],"colsample_bytree":[0.1,0.001,0.5]}
    # {"n_estimators":[90,110],"learning_rate":[0.1,0.001,0.5],
    #  "max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
    #  "colsample_bylevel":[0.6,0.7,0.9] }
]

#2. 모델구성 
model =RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist'),
                    parameters, cv=kfold,verbose=1,      
                    refit=True,n_jobs=-1)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 예측
result = model.score(x_test,y_test)
print(model,) 
print("결과:",result)
print("걸린시간",end-start)

# GridSearchCV
# 결과: 0.9676
# 걸린시간 20652.82614159584

# LinearDiscriminantAnalysis
# 결과: 0.9677
# 걸린시간 721.3816196918488