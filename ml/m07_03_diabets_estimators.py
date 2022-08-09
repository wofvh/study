from matplotlib.colors import rgb2hex
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
from sklearn.metrics import r2_score
#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

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
# r2 : [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369]
# cross_val_predict r2 : 0.49199169553792077
# r2 : [0.37384285 0.45740157 0.52494925 0.39481806 0.44036568]
# cross_val_predict r2 : 0.42253862477880355
# r2 : [0.39682022 0.47719997 0.46585017 0.31989962 0.3615444 ]
# cross_val_predict r2 : 0.3954603075477908
# r2 : [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ]
# cross_val_predict r2 : 0.48924312029289785
# r2 : [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701]
# cross_val_predict r2 : 0.4377873717445636
# r2 : [-0.2225803  -0.15250924 -0.1463291  -0.06850001  0.06343424]
# cross_val_predict r2 : -0.1354157344424367
# r2 : [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
#  -9.58335111e-03]
# cross_val_predict r2 : -0.0011952295491977072
# r2 : [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988]
# cross_val_predict r2 : 0.007459258657392942
# r2 : [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ]
# cross_val_predict r2 : 0.4390458408872532
# r2 : [-0.085248   -0.05810277 -0.20884251  0.00262604 -0.18328413]
# cross_val_predict r2 : -0.1518876343673623
# r2 : [0.36653344 0.45402078 0.49302717 0.40796685 0.45171467]
# cross_val_predict r2 : 0.4502270653060174
# r2 : [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898]
# cross_val_predict r2 : 0.005230437403623833
# r2 : [ -5.6360765  -15.27401277  -9.94981465 -12.46884533 -12.04795337]
# cross_val_predict r2 : -10.927861946750946
# r2 : [0.38790336 0.47789108 0.48172843 0.39414484 0.44284269]
# cross_val_predict r2 : 0.4385049026363981
# r2 : [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755]
# cross_val_predict r2 : 0.3908365808838361
# r2 : [0.50334678 0.47508239 0.54645899 0.36875267 0.51730207]
# cross_val_predict r2 : 0.482367102381571
# r2 : [nan nan nan nan nan]
# IsotonicRegression 은 안나온 놈
# r2 : [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969]
# cross_val_predict r2 : 0.36825446940886863
# r2 : [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537]
# cross_val_predict r2 : -3.570370718100797
# r2 : [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679]
# cross_val_predict r2 : -0.1091164656228727
# r2 : [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596]
# cross_val_predict r2 : 0.4876175223096061
# r2 : [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ]
# cross_val_predict r2 : 0.3523074646619849
# r2 : [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393]
# cross_val_predict r2 : 0.48674372242489805
# r2 : [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891]
# cross_val_predict r2 : 0.3745884219693264
# r2 : [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679]
# cross_val_predict r2 : 0.48631321449567955
# r2 : [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ]
# cross_val_predict r2 : 0.49091771918833926
# r2 : [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679]
# cross_val_predict r2 : 0.4876467758677233
# r2 : [-0.33470258 -0.31629592 -0.4189491  -0.30760952 -0.47389705]
# cross_val_predict r2 : -0.365438453198226
# r2 : [-2.77453973 -2.92761955 -3.37164868 -2.95080034 -3.07983356]
# cross_val_predict r2 : -3.0253286805821196
# MultiOutputRegressor 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskElasticNet 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskElasticNetCV 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskLasso 은 안나온 놈
# r2 : [nan nan nan nan nan]
# MultiTaskLassoCV 은 안나온 놈
# r2 : [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ]
# cross_val_predict r2 : 0.16261540190521795
# r2 : [0.32934491 0.285747   0.38943221 0.19671679 0.35916077]
# cross_val_predict r2 : 0.31250986991173957
# r2 : [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516]
# cross_val_predict r2 : 0.4851449098996172
# r2 : [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996]
# cross_val_predict r2 : -1.201061542714958
# r2 : [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873]
# cross_val_predict r2 : 0.4840765704541007
# r2 : [0.45818215 0.49064114 0.46602495 0.35526272 0.46137539]
# cross_val_predict r2 : 0.44998904880983537
# r2 : [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626]
# cross_val_predict r2 : 0.3436900047999266
# r2 : [ 0.26122316 -0.29002185  0.26417094  0.1635631   0.47007671]
# cross_val_predict r2 : -0.18710599194213295
# r2 : [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
#  -9.58335111e-03]
# cross_val_predict r2 : -0.0011952295491977072
# r2 : [0.35332545 0.50348386 0.48365815 0.41683544 0.40118058]
# cross_val_predict r2 : 0.43750709840884194
# RegressorChain 은 안나온 놈
# r2 : [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091]
# cross_val_predict r2 : 0.4207996370817283
# r2 : [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194]
# cross_val_predict r2 : 0.48831025738538514
# r2 : [0.39338035 0.44167346 0.46467051 0.32945716 0.41508668]
# cross_val_predict r2 : 0.40817041697962
# r2 : [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ]
# cross_val_predict r2 : 0.15989977853329007
# StackingRegressor 은 안나온 놈
# r2 : [0.50606713 0.4577781  0.55226714 0.32174153 0.53323212]
# cross_val_predict r2 : 0.47752017958241055
# r2 : [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679]
# cross_val_predict r2 : 0.4876467758677233
# r2 : [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042]
# cross_val_predict r2 : 0.005233807354719566
# VotingRegressor 은 안나온 놈