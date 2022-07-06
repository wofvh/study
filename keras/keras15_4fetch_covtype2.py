import numpy as np 
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf

import pandas as pd

#1.데이터 
datasets = fetch_covtype()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(581012, 54) (581012, )
print(np.unique(y,return_counts=True)) #(array[1 2 3 4 5 6 7],array[211840, 283301,  35754,   2747,   9493,  17367,  20510] )

#텐서플로우
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) 

# 판다스 겟더미
# y= pd.get_dummies(y) #argmax 다르게 하니 돌아감. 이유는 모름

#사이킷런
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y)

print(y)
print(y.shape)


#2.모델
model = Sequential()
model.add(Dense(10, input_dim=54))
model.add(Dense(10,activation ='relu'))
model.add(Dense(10,activation ='relu'))
model.add(Dense(10,activation ='relu'))
model.add(Dense(10,activation ='relu'))
model.add(Dense(7,activation ='softmax')) #소프트맥스는 모든 연산값의 합이 1.0,그중 가장 큰값(퍼센트)을 선택,so 마지막 노드3개* y의 라벨의 갯수
#softmax는 아웃풋만 가능 히든에서 x

#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58) #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=100,validation_split=0.2,callbacks=earlyStopping, verbose=1) #batch default :32


#4.평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
#################### 위와 동일###############
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

# print(y_test)
# print(y_test.shape)

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

print(y_predict)
print(y_test)


#겟더미
# loss :  0.6037192344665527
# acc :  0.7528570772902515
#사이킷런
# loss :  0.5900264382362366
# acc :  0.7511072608775473