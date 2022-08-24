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
y = train_set['Survived']
train_set = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = train_set
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron 
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

y = np.array(y).reshape(-1, 1)
print(x.shape,y.shape)              # (891, 7) (891, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8, shuffle=True, random_state=123)

y_train =np.array(y_train, dtype='float32')

x1 = tf.placeholder(tf.float32, shape=[None, 7])
y1 = tf.placeholder(tf.float32, shape=[None,1])
w = tf.compat.v1.Variable(tf.zeros([7,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias') 

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x1,w) + b)
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y_train*tf.log(hypothesis)+(1-y_train)*tf.log(1-hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20):
  
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x:x_train,y:y_train})
    if epochs %5 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', h_val)

#4. 예측

y_predict = sess.run(tf.cast(h_val>=0.5, dtype=tf.float32))   # 참이면 1 , 거짓이면 0
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error

print(y.shape,y_predict.shape) # (179, 1) (712, 1)

acc = sess.run(accuracy_score(y, y_predict))
print('acc : ', acc)

mse = mean_absolute_error(y_test, h_val)
print('mse : ', mse)
sess.close()


