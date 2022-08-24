from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
# from keras.layers.recurrent import  SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)# [10886 rows x 13 columns]
print(test_set)# [6493 rows x 12 columns]

##########################################

x_data = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
y_data = train_set['count'] 

print(y_data.shape)
# exit()

y_data = y_data.values.reshape(10886,1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.7, shuffle=True, random_state=72)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성 // 시작 
x = tf.compat.v1.placeholder(tf.float32,shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None, 1])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1],1],dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))

hypothesis = tf.matmul(x,w) +b

loss = tf.reduce_mean(tf.square(hypothesis-y))   
#loss = 'categorical_crossentropy'

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate=0.98).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(40001):
  
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x:x_train,y:y_train})
    if epochs %50 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', h_val)

#4. 예측
# y_predict =sess.run(tf.argmax(h_val))
# y_test = sess.run(tf.argmax(y_test))

# y_predict = sess.run(tf.cast(h_val>=0.5, dtype=tf.float32))   # 참이면 1 , 거짓이면 0
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error

y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_test.shape,y_predict.shape) # (179, 1) (712, 1)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# mse = mean_absolute_error(y, h_val)
# print('mse : ', mse)

sess.close()

# r2 :  0.39809930841493557



