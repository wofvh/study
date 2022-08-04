import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
datasets = load_breast_cancer()
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
# AdaBoostClassifier 의 정답률 : 0.9473684210526315
# BaggingClassifier 의 정답률 : 0.9649122807017544
# BernoulliNB 의 정답률 : 0.6403508771929824
# CalibratedClassifierCV 의 정답률 : 0.9649122807017544
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률 : 0.7807017543859649
# DecisionTreeClassifier 의 정답률 : 0.9210526315789473
# DummyClassifier 의 정답률 : 0.6403508771929824
# ExtraTreeClassifier 의 정답률 : 0.9122807017543859
# ExtraTreesClassifier 의 정답률 : 0.9649122807017544
# GaussianNB 의 정답률 : 0.9210526315789473
# GaussianProcessClassifier 의 정답률 : 0.9649122807017544
# GradientBoostingClassifier 의 정답률 : 0.9473684210526315
# HistGradientBoostingClassifier 의 정답률 : 0.9736842105263158
# KNeighborsClassifier 의 정답률 : 0.956140350877193
# LabelPropagation 의 정답률 : 0.9473684210526315
# LabelSpreading 의 정답률 : 0.9473684210526315
# LinearDiscriminantAnalysis 의 정답률 : 0.9473684210526315
# LinearSVC 의 정답률 : 0.9736842105263158
# LogisticRegression 의 정답률 : 0.9649122807017544
# LogisticRegressionCV 의 정답률 : 0.9736842105263158
# MLPClassifier 의 정답률 : 0.9649122807017544
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률 : 0.8508771929824561
# NearestCentroid 의 정답률 : 0.9298245614035088
# NuSVC 의 정답률 : 0.9473684210526315
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률 : 0.9736842105263158
# Perceptron 의 정답률 : 0.9736842105263158
# QuadraticDiscriminantAnalysis 의 정답률 : 0.9385964912280702
# RadiusNeighborsClassifier 은 안나온 놈
# RandomForestClassifier 의 정답률 : 0.9649122807017544
# RidgeClassifier 의 정답률 : 0.9473684210526315
# RidgeClassifierCV 의 정답률 : 0.9473684210526315
# SGDClassifier 의 정답률 : 0.9736842105263158
# SVC 의 정답률 : 0.9736842105263158
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈