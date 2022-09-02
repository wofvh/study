from sklearn import datasets
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.set_random_seed(123)
import math
import pandas as pd
#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')             # index_col=n n번째 컬럼을 인덱스로 인식
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)   
print(train_set['Embarked'].mode())  # 0    S / Name: Embarked, dtype: object
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)                     # mode 모르겠다..
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)  # replace 교체하겠다.
y_data = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x_data = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

y_data = np.array(y_data).reshape(-1, 1)
# print(x.shape,y.shape)              # (891, 7) (891, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=123,stratify=y_data)

# y_train =np.array(y_train, dtype='float32')

#2. 모델구성 // 시작 
x = tf.compat.v1.placeholder(tf.float32,shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None, 1])
w = tf.compat.v1.Variable(tf.compat.v1.zeros([x_data.shape[1],1],dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1],dtype=tf.float32))

###############################################################
# w1 =tf.compat.v1.Variable(tf.random_normal([2, 20]))
# b1= tf.compat.v1.Variable(tf.random_normal([20]))

h1 = tf.matmul(x,w)+b

w2 =tf.compat.v1.Variable(tf.random_normal([y_data.shape[1], 30]))
b2= tf.compat.v1.Variable(tf.random_normal([30]))

h2 =tf.nn.sigmoid(tf.matmul(h1,w2)+b2)

w3 =tf.compat.v1.Variable(tf.random_normal([30, 20]))
b3= tf.compat.v1.Variable(tf.random_normal([20]))

h3 =tf.nn.sigmoid(tf.matmul(h2,w3)+b3)

#output layer
w4 = tf.compat.v1.Variable(tf.random_normal([20, 1]))
b4= tf.compat.v1.Variable(tf.random_normal([1]))

hypothesis = tf.nn.sigmoid(tf.matmul(h3,w4) +b4)

#############################################

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate= 1e-6).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2000):
  
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
y_predict = sess.run(tf.cast(y_predict > 0.5, dtype=tf.float32))

print(y_test.shape,y_predict.shape) # (179, 1) (712, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)

# mse = mean_absolute_error(y, h_val)
# print('mse : ', mse)

sess.close()


# accuracy_score :  0.6983240223463687
# accuracy_score :  0.6145251396648045