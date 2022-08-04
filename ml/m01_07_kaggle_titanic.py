from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import math

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC

y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

#2. 모델
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

# model = LinearSVC()
# 결과 : 0.7486033519553073
# acc스코어 :  0.7486033519553073