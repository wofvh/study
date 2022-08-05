from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import math

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

#2. 모델
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)
#2. 모델
model =  RandomForestClassifier()

from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') 
from sklearn.preprocessing import MinMaxScaler


#2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms:',allAlgorithms)
print('모델개수:',len(allAlgorithms))

for (name,algorithm) in  allAlgorithms :
    try :
        model = algorithm()

        scores = cross_val_score(model,x, y,cv=kfold)
        print('cross_val_score :' ,scores)
    except:
        # continue
        print(name,"은 안나온 놈")
        
# 모델개수: 41
# cross_val_score : [0.81005587 0.81460674 0.80337079 0.79213483 0.82022472]
# cross_val_score : [0.80446927 0.80898876 0.79213483 0.75280899 0.80898876]
# cross_val_score : [0.77094972 0.78651685 0.78651685 0.78089888 0.79213483]
# cross_val_score : [0.80446927 0.8258427  0.7752809  0.7752809  0.82022472]
# cross_val_score : [0.77094972        nan 0.78089888 0.7752809         nan]
# ClassifierChain 은 안나온 놈
# cross_val_score : [0.67039106 0.7247191  0.70786517 0.73033708 0.64044944]
# cross_val_score : [0.76536313 0.78089888 0.75842697 0.74719101 0.79213483]
# cross_val_score : [0.61452514 0.61797753 0.61797753 0.61797753 0.61235955]
# cross_val_score : [0.76536313 0.78089888 0.75842697 0.70224719 0.76404494]
# cross_val_score : [0.77094972 0.78651685 0.76966292 0.75842697 0.80337079]
# cross_val_score : [0.77653631 0.80337079 0.78651685 0.75280899 0.80898876]
# cross_val_score : [0.73743017 0.70224719 0.71348315 0.68539326 0.71348315]
# cross_val_score : [0.79888268 0.85393258 0.80337079 0.79775281 0.8258427 ]
# cross_val_score : [0.81005587 0.83146067 0.80337079 0.78089888 0.8258427 ]
# cross_val_score : [0.74301676 0.69101124 0.69101124 0.6741573  0.68539326]
# cross_val_score : [0.72067039 0.67977528 0.69101124 0.64606742 0.73033708]
# cross_val_score : [0.72067039 0.67977528 0.69101124 0.64606742 0.73033708]
# cross_val_score : [0.79888268 0.79213483 0.79213483 0.78651685 0.79775281]
# cross_val_score : [0.60335196 0.51685393 0.67977528 0.75280899 0.83146067]
# cross_val_score : [0.82122905 0.7752809  0.79775281 0.78089888 0.81460674]
# cross_val_score : [0.82122905 0.7752809  0.78651685 0.78089888 0.81460674]
# cross_val_score : [0.79888268 0.82022472 0.80337079 0.76404494 0.80337079]
# MultiOutputClassifier 은 안나온 놈
# cross_val_score : [0.67039106 0.7247191  0.70786517 0.71910112 0.64044944]
# cross_val_score : [0.66480447 0.66853933 0.70786517 0.67977528 0.61235955]
# cross_val_score : [0.78212291 0.82022472 0.80898876 0.74157303 0.8258427 ]
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# cross_val_score : [0.65363128 0.63483146 0.75842697 0.63483146 0.56741573]
# cross_val_score : [0.6424581  0.62921348 0.62359551 0.71348315 0.64044944]
# cross_val_score : [0.78212291 0.8258427  0.80337079 0.74719101 0.82022472]
# cross_val_score : [nan nan nan nan nan]
# cross_val_score : [0.79888268 0.80898876 0.78089888 0.76966292 0.8258427 ]
# cross_val_score : [0.79888268 0.79213483 0.79213483 0.78651685 0.79775281]
# cross_val_score : [0.79888268 0.79213483 0.79213483 0.78651685 0.79775281]
# cross_val_score : [0.80446927 0.80337079 0.30898876 0.6741573  0.42134831]
# cross_val_score : [0.67039106 0.71910112 0.6741573  0.69101124 0.61797753]
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈