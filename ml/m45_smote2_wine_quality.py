import pandas as pd
# 11번째까지 x 나머지 y 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel   # 모델을 선택.
from imblearn.over_sampling import SMOTE  # SMOTE install 필요

#1. 데이터
data = pd.read_csv('C:\study\_data\wine/winequality-white.csv',header=0,sep=';')

print(data.shape)
print(data.describe()) 

x = data.values[:,0:11]
y = data.values[:,11]


x = x[:-3]
y = y[:-3]
# print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    8

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123,
                                                    shuffle=True,
                                                    train_size=0.75,
                                                    stratify=y
                                                    )

print(pd.Series(y_train).value_counts())
# 1    57
# 0    47
# 2     6
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.neighbors import KNeighborsClassifier

# 모델 생성
# model = KNeighborsClassifier(n_neighbors=3)



model.fit(x_train,y_train)

#4. 평가
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
# print('결과:',score)
print('acc:', accuracy_score(y_test,y_predict))
print('f1_macro:',f1_score(y_test,y_predict, average='macro'))      # 이진분류 f1스코어 프리시즌(재현) 리콜(정밀도) 
# print('f1_micro:',f1_score(y_test,y_predict, average='micro'))  

# 기본
# acc: 0.9722222222222222
# f1_macro: 0.9743209876543211

# 30 개 축소
# acc: 0.9666666666666667
# f1_macro: 0.9743209876543211

# 40 개 축소
# acc: 0.9285714285714286
# f1_macro: 0.8517460317460318

print('=========================smote 적용 후==========================')
smote = SMOTE(random_state=123,k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train,y_train)


print(pd.Series(y_train).value_counts())
# 0    53                           # smote 값에 개수를 맞춰줌. 단점 : 증폭방식이 제곱여서 수치가 커질 수록 속도가 느려진다. 
# 1    53
# 2    53


# 모델,훈련
model = RandomForestClassifier()
model.fit(x_train,y_train)

#4. 평가
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
print('acc:', accuracy_score(y_test,y_predict))
print('f1_macro:',f1_score(y_test,y_predict, average='macro'))  