import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score



#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

#2. 모델
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression     # LogisticRegression 분류모델
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier   # >Regression 회귀모델은 변경 


model = LinearSVC()

#3. 컴파일,훈련
model.fit(x_train,y_train)

#4. 평가, 예측

results = model.score(x_test,y_test)   # = evaluate 
print("결과 :",results)                 # 회귀는 = r2스코어 분류는 acc 값과 동일. 

y_predict = model.predict(x_test)

acc= accuracy_score(y_test, y_predict) 
print('acc스코어 : ', acc) 

# loss :  0.0530550517141819
# accuracy :  1.0