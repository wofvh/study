import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer,load_wine, load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

#1. 데이터
datasets = load_digits()

# df = pd.DataFrame(datasets.data,columns=datasets.feature_names)

# print(df.head(7))

x_train, x_test,y_train,y_test = train_test_split (datasets.data,datasets.target,train_size=0.8,random_state=123,shuffle=True,stratify=datasets.target)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 .모델

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
lgb = LGBMClassifier()
cvb = CatBoostClassifier(verbose=0)
xg = XGBClassifier()


model = VotingClassifier(estimators=[('LR',lr),('KNN',knn),('LGB',lgb),('CVB',cvb),('XG',xg)],
                         voting='soft'         #hard 여러모델 중에 투표하여 결정 soft 각 모델에 대해 퍼센트지가 높은 것을 선택하여 출력.
                         )

#3. 훈련
model.fit(x_train,y_train)

#4. 평가
y_predict = model.predict(x_test)

score = accuracy_score(y_test,y_predict)
print('voting 결과',round(score,4))

# 결과 0.9912

classifiers = [lr,knn,lgb,cvb,xg]
for model2 in classifiers :
    model2.fit(x_train,y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_predict)
    class_name =model2.__class__.__name__
    print('{0} 정확도:{1:.4f}'.format(class_name,score2))
    
    
# voting 결과 0.9917
# LogisticRegression 정확도:0.9750
# KNeighborsClassifier 정확도:0.9778
# LGBMClassifier 정확도:0.9778
# CatBoostClassifier 정확도:0.9917
# XGBClassifier 정확도:0.9750


