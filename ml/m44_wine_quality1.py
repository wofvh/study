import pandas as pd
# 11번째까지 x 나머지 y 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel   # 모델을 선택.

#1. 데이터
data = pd.read_csv('C:\study\_data\wine/winequality-white.csv',header=0,sep=';')

print(data.shape)

# "quality"

# x = np.array(data.drop['quality'])
# y = np.array(data['quality'])

x = data.values[:,0:11]
y = data.values[:,11]

print(x.shape,y.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=123
                                                    ,stratify=y)
# random_state로 값이 변화하는이유는 분포가 골고루 되지 않았았기 때문이다. 

from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor        # activate tf282gpu > pip install xgboost 

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()
    # random_state=123,                 #위에있는 파라미터를 모델안에 넣을때 하는 방법
    #                 n_estimators=100,
    #                 learning_rate=0.1,
    #                 max_depth=3,
    #                 gamma=1)

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train, early_stopping_rounds =10,
          eval_set = [(x_train,y_train),(x_test,y_test)],   # 훈련 + 학습 # 뒤에걸 인지한다
        #   eval_set = [(x_test,y_test)]                 
        eval_metric='merror',          
          # rmse,mae,mrmsle...  회귀             
          # errror, auc...      이진  
          # merror,mlogloss..   다중
          )

#4. 예측

from sklearn.metrics import accuracy_score, r2_score

result = model1.score(x_test,y_test)
print("model.score:",result)

y_predict = model1.predict(x_test)
acc = accuracy_score(y_test,y_predict)

print( 'accuracy_score :',acc)
print(model1,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")


result2 = model2.score(x_test,y_test)
print("model2.score:",result2)

y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test,y_predict2)

print( 'accuracy2_score :',acc2)
print(model2,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")


result3 = model3.score(x_test,y_test)
print("model3.score:",result3)

y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test,y_predict3)

print( 'accuracy3_score :',acc3)
print(model3,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

result4 = model4.score(x_test,y_test)
print("model4.score:",result4)

y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test,y_predict3)

print( 'accuracy4_score :',acc4)
print(model4,':')   # 중요한 피쳐를 구분하는 것 중요성이 떨어지는것을 버린다. 
print("===================================")

# ===================================
# model2.score: 0.7438775510204082
# accuracy2_score : 0.7438775510204082
# RandomForestClassifier() :

print(model2.feature_importances_)
# [0.07582085 0.09918254 0.08239677 0.08706568 0.08501132 0.0940047
#  0.09110613 0.10354693 0.08669432 0.0808266  0.11434415]

thresholds =model2.feature_importances_
print('=========================================')
for thresh in thresholds:
    selection = SelectFromModel(model2, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    print(select_x_train.shape,select_x_test.shape)

    selection_model = XGBClassifier(random_state=123,                 #위에있는 파라미터를 모델안에 넣을때 하는 방법
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    gamma=1)
    
    selection_model.fit(select_x_train,y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test,y_predict)
    
    print('Thresh=%.3f, n=%d, acc: %.2f%%'
          %(thresh, select_x_train.shape[1],score*100))
    
    