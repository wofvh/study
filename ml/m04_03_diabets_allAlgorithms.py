import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
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
datasets = load_diabetes()
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
        
# 모델개수: 41
# AdaBoostClassifier 의 정답률 : 0.0
# BaggingClassifier 의 정답률 : 0.011235955056179775
# BernoulliNB 의 정답률 : 0.0
# CalibratedClassifierCV 의 정답률 : 0.0
# CategoricalNB 의 정답률 : 0.0
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률 : 0.02247191011235955
# DecisionTreeClassifier 의 정답률 : 0.02247191011235955
# DummyClassifier 의 정답률 : 0.0
# ExtraTreeClassifier 의 정답률 : 0.02247191011235955
# ExtraTreesClassifier 의 정답률 : 0.02247191011235955
# GaussianNB 의 정답률 : 0.011235955056179775
# GaussianProcessClassifier 의 정답률 : 0.011235955056179775
# GradientBoostingClassifier 의 정답률 : 0.02247191011235955
# HistGradientBoostingClassifier 의 정답률 : 0.02247191011235955
# KNeighborsClassifier 의 정답률 : 0.0
# LabelPropagation 의 정답률 : 0.011235955056179775
# LabelSpreading 의 정답률 : 0.011235955056179775
# LinearDiscriminantAnalysis 의 정답률 : 0.0
# LinearSVC 의 정답률 : 0.0
# LogisticRegression 의 정답률 : 0.0
# LogisticRegressionCV 은 안나온 놈
# MLPClassifier 의 정답률 : 0.011235955056179775
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률 : 0.0
# NearestCentroid 의 정답률 : 0.011235955056179775
# NuSVC 은 안나온 놈
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률 : 0.0
# Perceptron 의 정답률 : 0.0
# QuadraticDiscriminantAnalysis 은 안나온 놈
# RadiusNeighborsClassifier 의 정답률 : 0.0
# RandomForestClassifier 의 정답률 : 0.02247191011235955
# RidgeClassifier 의 정답률 : 0.0
# RidgeClassifierCV 의 정답률 : 0.0
# SGDClassifier 의 정답률 : 0.0
# SVC 의 정답률 : 0.011235955056179775
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈        