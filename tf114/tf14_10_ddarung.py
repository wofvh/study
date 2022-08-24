from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 # pandas : 엑셀땡겨올때 씀 python 지원하는 엑셀을 불러오는 기능.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import tensorflow as tf
#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)

train_set = train_set.fillna(train_set.mean())       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값

x_data = train_set.drop(['count'], axis=1)                    # drop 데이터에서 ''사이 값 빼기

y_data = train_set['count'] 
print(y_data.shape)             # (1459,)

y_data = y_data.values.reshape(1459,1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.9, shuffle=True, random_state=72)

print(x_train.shape)        # (1313, 9)
print(y_train.shape)        # (1313,)
print(x_test.shape)         # (146, 9)
print(y_test.shape)         # (146,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# y_train =np.array(y_train, dtype='float32')

#2. 모델구성 // 시작 
x = tf.compat.v1.placeholder(tf.float32,shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None, 1])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1],1],dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))

hypothesis = tf.matmul(x,w) +b

loss = tf.reduce_mean(tf.square(hypothesis-y))   
#loss = 'categorical_crossentropy'

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate=0.9).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(200):
  
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

# r2 : 0.6042134111979978



