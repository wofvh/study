from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#1. 데이터 
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

print(x.shape,y.shape)  # (569, 30) (569,)


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=123, shuffle=True,stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression         # sigmoid 
from sklearn.ensemble import BaggingClassifier               
# bagging할 때 scaler를 해야한다. 한가지모델을 여러번 돌린다.(안에 파라미터를 다르게한다)
# 1. 붓스트랩 샘플링: 데이터 샘플은 복원추출(랜덤포레스트와 같다)
# 2. 분류기를 트리외에 다양하게 쓸수 있는 앙상블 기법
# (예를들어, 로지스틱 회귀로 100번 써서 강화하는등)


#2. 모델 
model = BaggingClassifier(LogisticRegression(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=123,
                          )
#3. 훈련 
model.fit(x_train,y_train)



#4. 예측,평가
print(model.score(x_test,y_test))






