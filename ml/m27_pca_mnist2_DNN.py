
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold

(x_train,y_train),(x_test,y_test) =mnist.load_data()       # _를 사용하면 사용하지않겠다. 
print(x_train.shape,x_test.shape)   # (60000, 28, 28) (10000, 28, 28)
 
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000,784)

pca = PCA(n_components=154)   
x= pca.fit_transform(x) 

pca_EVR = pca.explained_variance_ratio_ 
cumsum = np.cumsum(pca_EVR)

x_train,x_test,y_train,y_test = train_test_split(x,y , train_size=0.8, random_state=123, shuffle=True) 
# # x = np.append(x_train,x_test,axis=0)
# # print(x.shape)                      # (70000, 28, 28)

# x_train= x_train.reshape(60000, 28*28)   
# x_test= x_test.reshape(10000, 28*28)   

# pca = PCA(n_components=154)               # 주성분 / 열축소 13 > 2개로 압축. 
# x = pca.fit_transform(x_train)
# # y = pca.transform(y_train)

# pca_EVR = pca.explained_variance_ratio_                      
# cumsum = np.cumsum(pca_EVR)             

print(np.argmax(cumsum >=0.95) + 1)         # 154
print(np.argmax(cumsum >=0.99) + 1)         # 331
print(np.argmax(cumsum >=0.999) + 1)        # 486
print(np.argmax(cumsum >=1.0) + 1)          # 713

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

# mode = DecisionTreeClassifier()
model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()
#4. 예측

from sklearn.metrics import accuracy_score, r2_score

result = model.score(x_test,y_test)
print("model.score:",result)
print("걸린시간",end-start)
print(model,)   
print("===================================")



#[실습]
# 아까 4가지로 모델을 만들기 
# 784개 DNN으로 만든거 (최상의 성능인거 //0.993이상)

# time 체크 / fit에서 하고 

# 1. 나의 최고 DNN
# loss :  0.3970048725605011
# acc :  0.9568
# 걸린시간 :  146.47142386436462

# 2. 나의 최고 CNN
# acc :  0.986
# 걸린시간: 77.1074709892273

# 3. PCA 0.95
# model.score: 0.9707
# accuracy_score : 0.9707
# RandomForestClassifier() :
# ===================================
# 걸린시간 35.764737606048584

# 4. PCA 0.99
# model.score: 0.9694
# accuracy_score : 0.9694
# RandomForestClassifier() :
# ===================================
# 걸린시간 36.09515190124512

# 5. PCA 0.999
# model.score: 0.9696
# accuracy_score : 0.9696
# RandomForestClassifier() :
# ===================================
# 걸린시간 36.197712421417236

# 6. PCA 1.0
# model.score: 0.9704
# accuracy_score : 0.9704
# RandomForestClassifier() :
# ===================================
# 걸린시간 35.87223172187805
