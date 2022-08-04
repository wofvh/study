import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from yaml import warnings
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
import tensorflow as tf



#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
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
# AdaBoostClassifier 의 정답률 : 0.6333333333333333
# BaggingClassifier 의 정답률 : 0.9333333333333333
# BernoulliNB 의 정답률 : 0.4
# CalibratedClassifierCV 의 정답률 : 0.9666666666666667
# CategoricalNB 의 정답률 : 0.3333333333333333
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률 : 0.6666666666666666
# DecisionTreeClassifier 의 정답률 : 0.9666666666666667
# DummyClassifier 의 정답률 : 0.3
# ExtraTreeClassifier 의 정답률 : 0.8666666666666667
# ExtraTreesClassifier 의 정답률 : 0.9666666666666667
# GaussianNB 의 정답률 : 0.9666666666666667
# GaussianProcessClassifier 의 정답률 : 0.9666666666666667
# GradientBoostingClassifier 의 정답률 : 0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 : 0.8666666666666667
# KNeighborsClassifier 의 정답률 : 1.0
# LabelPropagation 의 정답률 : 0.9666666666666667
# LabelSpreading 의 정답률 : 0.9666666666666667
# LinearDiscriminantAnalysis 의 정답률 : 1.0
# LinearSVC 의 정답률 : 0.9666666666666667
# LogisticRegression 의 정답률 : 0.9666666666666667
# LogisticRegressionCV 의 정답률 : 1.0
# MLPClassifier 의 정답률 : 0.9
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률 : 0.6333333333333333
# NearestCentroid 의 정답률 : 0.9666666666666667
# NuSVC 의 정답률 : 0.9666666666666667
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률 : 0.7
# Perceptron 의 정답률 : 0.9333333333333333
# QuadraticDiscriminantAnalysis 의 정답률 : 1.0
# RadiusNeighborsClassifier 의 정답률 : 0.4666666666666667
# RandomForestClassifier 의 정답률 : 0.9333333333333333
# RidgeClassifier 의 정답률 : 0.9333333333333333
# RidgeClassifierCV 의 정답률 : 0.8333333333333334
# SGDClassifier 의 정답률 : 0.9
# SVC 의 정답률 : 1.0
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈

# Perceptron : 0.9333333333333333
# LogisticRegression : 1.0
# KNeighborsClassifier : 0.9666666666666667
# DecisionTreeClassifier : 0.9666666666666667        
# RandomForestClassifier : 0.9333333333333333