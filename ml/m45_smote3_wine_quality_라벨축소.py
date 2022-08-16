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


x = x[:-6]
y = y[:-6]
print(pd.Series(y).value_counts())
# 6.0    2196
# 5.0    1457
# 7.0     879
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5


newlist = []
for i in y:
    if i <5: 
        newlist += [0]
    elif i < 6:
        newlist += [1] 
    elif i < 7:
        newlist += [2] 
    else:
        newlist += [3]
               
print(np.unique(newlist,return_counts=True))  

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123,
                                                    shuffle=True,
                                                    train_size=0.75,
                                                    stratify=y
                                                    )

print(pd.Series(y_train).value_counts())
print('---------------')
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

# acc: 0.7252657399836467
# f1_macro: 0.4278355346494283
# =========================smote 적용 후==========================
# 6.0    1646
# 8.0    1646
# 7.0    1646
# 5.0    1646
# 4.0    1646
# 3.0    1646
# 9.0    1646
# dtype: int64
# acc: 0.6770237121831562
# f1_macro: 0.4247564978896075

