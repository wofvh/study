import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf


from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

#2. 모델
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms:',allAlgorithms)
print('모델개수:',len(allAlgorithms))

for (name,algorithm) in  allAlgorithms :
    try :
        model = algorithm()
        model.fit(x_train,y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name,'의 정답률 :',acc )
    except:
        # continue
        print(name,"은 안나온 놈")
    

# ah델개수: 41
# AdaBoostClassifier 의 정답률 : 0.8888888888888888
# BaggingClassifier 의 정답률 : 1.0
# BernoulliNB 의 정답률 : 0.4166666666666667
# CalibratedClassifierCV 의 정답률 : 0.9722222222222222
# CategoricalNB 의 정답률 : 0.5
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률 : 0.8611111111111112
# DecisionTreeClassifier 의 정답률 : 0.9444444444444444
# DummyClassifier 의 정답률 : 0.4166666666666667
# ExtraTreeClassifier 의 정답률 : 0.8333333333333334
# ExtraTreesClassifier 의 정답률 : 1.0
# GaussianNB 의 정답률 : 1.0
# GaussianProcessClassifier 의 정답률 : 1.0
# GradientBoostingClassifier 의 정답률 : 0.9722222222222222
# HistGradientBoostingClassifier 의 정답률 : 0.9722222222222222
# KNeighborsClassifier 의 정답률 : 1.0
# LabelPropagation 의 정답률 : 1.0
# LabelSpreading 의 정답률 : 1.0
# LinearDiscriminantAnalysis 의 정답률 : 1.0
# LinearSVC 의 정답률 : 0.9722222222222222
# LogisticRegression 의 정답률 : 1.0
# LogisticRegressionCV 의 정답률 : 0.9722222222222222
# MLPClassifier 의 정답률 : 1.0
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률 : 0.9444444444444444
# NearestCentroid 의 정답률 : 1.0
# NuSVC 의 정답률 : 1.0
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률 : 0.9722222222222222
# Perceptron 의 정답률 : 0.9722222222222222
# QuadraticDiscriminantAnalysis 의 정답률 : 0.9722222222222222
# RadiusNeighborsClassifier 의 정답률 : 0.9722222222222222
# RandomForestClassifier 의 정답률 : 1.0
# RidgeClassifier 의 정답률 : 1.0
# RidgeClassifierCV 의 정답률 : 0.9722222222222222
# SGDClassifier 의 정답률 : 1.0
# SVC 의 정답률 : 1.0
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈