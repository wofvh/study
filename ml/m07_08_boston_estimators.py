# 과제
# ativation : sigmoid, relu, linear
# metrics 추가 
# earlystopping 포함.
# 성능비교
# 감상문 2줄이상 
# 구글원격 
# r2값? loss값 ? accuracy값? 
# california , diabet, boston >> 회귀모델 metrics=mse, mae 값 프린트 (relu 1.2,3 사용할 때마다 뭐가 다른지)


import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import r2_score
#1. 데이터

datasets = load_boston()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model =  RandomForestClassifier()

from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') 
from sklearn.preprocessing import MinMaxScaler


#2. 모델
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms:',allAlgorithms)
print('모델개수:',len(allAlgorithms))

for (name,algorithm) in  allAlgorithms :
    try :
        model = algorithm()

        scores = cross_val_score(model,x, y,cv=kfold)
        print('r2 :' ,scores)
        y_predict = cross_val_predict(model,x, y,cv=kfold)
        r2 =r2_score(y,y_predict)
        print('cross_val_predict r2 :', r2 )
    except:
        # continue
        print(name,"은 안나온 놈")
        
# 모델개수: 54
# r2 : [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866]
# cross_val_predict r2 : 0.7039899224081672
# r2 : [0.90289436 0.80901382 0.80183585 0.83534738 0.89092235]
# cross_val_predict r2 : 0.8425167187615883
# r2 : [0.8923352  0.8049188  0.8209266  0.86711158 0.88557897]
# cross_val_predict r2 : 0.8620880593046795
# r2 : [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051]
# cross_val_predict r2 : 0.7080417384232369
# r2 : [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276]
# cross_val_predict r2 : 0.657552648539524
# r2 : [0.80253616 0.67211397 0.77164227 0.74656085 0.80791493]
# cross_val_predict r2 : 0.7640427405496409
# r2 : [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
# cross_val_predict r2 : -0.004749655056597302
# r2 : [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354]
# cross_val_predict r2 : 0.67339343815248
# r2 : [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608]
# cross_val_predict r2 : 0.659022239314722
# r2 : [0.74567471 0.73569275 0.55702629 0.70935208 0.67712052]
# cross_val_predict r2 : 0.6717809944944283
# r2 : [0.93570279 0.8571226  0.77350053 0.88000585 0.93266545]
# cross_val_predict r2 : 0.8790227203039956
# r2 : [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635]
# cross_val_predict r2 : -0.004749655056597302
# r2 : [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828]
# cross_val_predict r2 : -5.851764805978412
# r2 : [0.94613089 0.83343495 0.82760968 0.8868431  0.93147316]
# cross_val_predict r2 : 0.8889579317915464
# r2 : [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226]
# cross_val_predict r2 : 0.8616974518937098
# r2 : [0.70881407 0.65909542 0.53203819 0.36322935 0.62953938]
# cross_val_predict r2 : 0.5824536156078989
# r2 : [nan nan nan nan nan]
# IsotonicRegression 은 안나온 놈
# r2 : [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856]
# cross_val_predict r2 : 0.5265272092652364
# r2 : [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555]
# cross_val_predict r2 : 0.6913179708117978
# r2 : [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384]
# cross_val_predict r2 : 0.7013736501825939
# r2 : [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854]
# cross_val_predict r2 : 0.6974301488398669
# r2 : [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473]
# cross_val_predict r2 : 0.6681054172935808
# r2 : [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127]
# cross_val_predict r2 : 0.6804727762169174
# r2 : [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
# cross_val_predict r2 : -0.004749655056597302
# r2 : [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787]
# cross_val_predict r2 : 0.7014618041711322
# r2 : [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009]
# cross_val_predict r2 : 0.7175741349360552
# r2 : [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
# cross_val_predict r2 : 0.7173865443531537
# r2 : [ 0.48348837  0.39116501  0.43858102 -0.25148473 -0.13989109]
# cross_val_predict r2 : 0.263705825490842
# r2 : [0.46121154 0.45557958 0.32531858 0.37868203 0.38508473]
# cross_val_predict r2 : 0.4925813274731342
# MultiOutputRegressor 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskElasticNet 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskElasticNetCV 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskLasso 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskLassoCV 은 안나온 놈
# r2 : [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ]
# cross_val_predict r2 : 0.23179665043248865
# r2 : [0.58276176 0.565867   0.48689774 0.51545117 0.52049576]
# cross_val_predict r2 : 0.5386942259730013
# r2 : [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377]
# cross_val_predict r2 : 0.663156033793626
# r2 : [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868]
# cross_val_predict r2 : -2.1434902804905143
# r2 : [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313]
# cross_val_predict r2 : 0.6915588009621582
# r2 : [-1.84628975  0.0108803   0.08861087  0.13011108  0.30873937]
# cross_val_predict r2 : -2.23386212950571
# r2 : [0.85570647 0.81899779 0.66801489 0.67994598 0.7670857 ]
# cross_val_predict r2 : 0.7667757825037363
# r2 : [0.60032842 0.58307692 0.565067   0.42559134 0.65154092]
# cross_val_predict r2 : 0.3461314045453493
# r2 : [nan nan nan nan nan]
# RadiusNeighborsRegressor 은 안나온 놈
# r2 : [0.9170039  0.83985425 0.8212829  0.88170898 0.9055681 ]
# cross_val_predict r2 : 0.8787385616033857
# RegressorChain 은 안나온 놈
# r2 : [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776]
# cross_val_predict r2 : 0.7155708224424309
# r2 : [0.81125292 0.80010535 0.58888303 0.64008984 0.72362912]
# cross_val_predict r2 : 0.7173991519423358
# r2 : [-3.39621569e+25 -2.32828352e+26 -3.70119922e+26 -2.37292798e+26
#  -1.29071005e+25]
# cross_val_predict r2 : -3.266907740665389e+26
# r2 : [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554]
# cross_val_predict r2 : 0.19832004306905227
# StackingRegressor 은 안나온 놈
# r2 : [0.77430746 0.74085972 0.59879353 0.55301012 0.71624279]
# cross_val_predict r2 : 0.6780923730258765
# r2 : [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
# cross_val_predict r2 : 0.7173865443531537
# r2 : [0.7320775  0.75549621 0.57408841 0.57661534 0.63094693]
# cross_val_predict r2 : 0.6567515688229087
# VotingRegressor 은 안나온 놈