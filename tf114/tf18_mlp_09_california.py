from os import X_OK
from sklearn import datasets
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston, fetch_california_housing
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.set_random_seed(66)

#1. 데이터
datasets = fetch_california_housing()
x_data = datasets.data
y_data = datasets.target

y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.9, shuffle=True, random_state=72)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# y_train =np.array(y_train, dtype='float32')

#2. 모델구성 // 시작 
x = tf.compat.v1.placeholder(tf.float32,shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32,shape=[None, 1])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1],1],dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))

###############################################################
# w1 =tf.compat.v1.Variable(tf.random_normal([2, 20]))
# b1= tf.compat.v1.Variable(tf.random_normal([20]))

h1 = tf.matmul(x,w)+b

w2 =tf.compat.v1.Variable(tf.random_normal([y_data.shape[1], 30]))
b2= tf.compat.v1.Variable(tf.random_normal([30]))

h2 =tf.matmul(h1,w2)+b2

w3 =tf.compat.v1.Variable(tf.random_normal([30, 20]))
b3= tf.compat.v1.Variable(tf.random_normal([20]))

h3 =tf.matmul(h2,w3)+b3

#output layer
w4 = tf.compat.v1.Variable(tf.random_normal([20, 1]))
b4= tf.compat.v1.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(h3,w4) +b4

#############################################

loss = tf.reduce_mean(tf.square(hypothesis-y))   
#loss = 'categorical_crossentropy'

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

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

# r2 :  0.5448508484900801

#mlp 후
# r2 :  0.6078650998415176



