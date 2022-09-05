import imp
import tensorflow as tf
import keras
import numpy as np


tf.compat.v1.set_random_seed(123)


#1. 데이터

from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) =mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32,shape=[None, 28*28])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([28*28,10]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]))

y = tf.compat.v1.placeholder(tf.float32,shape=[None, 10])
###############################################################
# w1 =tf.compat.v1.Variable(tf.random_normal([2, 20]))
# b1= tf.compat.v1.Variable(tf.random_normal([20]))

h1 = tf.matmul(x,w)+b

w2 =tf.compat.v1.Variable(tf.random_normal([10, 28]))
b2= tf.compat.v1.Variable(tf.random_normal([28]))

h2 =tf.nn.sigmoid(tf.matmul(h1,w2)+b2)

w3 =tf.compat.v1.Variable(tf.random_normal([28, 20]))
b3= tf.compat.v1.Variable(tf.random_normal([20]))

h3 =tf.nn.sigmoid(tf.matmul(h2,w3)+b3)

#output layer
w4 = tf.compat.v1.Variable(tf.random_normal([20, 10]))
b4= tf.compat.v1.Variable(tf.random_normal([10]))

hypothesis = tf.nn.softmax(tf.matmul(h3,w4) +b4)

#############################################

# loss = tf.reduce_mean(tf.reduce_sum(y*tf.log(hypothesis),axis=1))  
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=tf.matmul(x,w) +b)  

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

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
# y_predict = sess.run(tf.cast(y_predict > 0.5, dtype=tf.float32))

print(y_test.shape,y_predict.shape) # (179, 1) (712, 1)
y_predict = sess.run(tf.argmax(y_predict, axis=1))
y_test = sess.run(tf.argmax(y_test, axis=1))

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)

# mse = mean_absolute_error(y, h_val)
# print('mse : ', mse)

sess.close()

# model = Sequential()
# # model.add(Dense(64, input_shape =(28*28, )))
# # # model.add(Dense(64, input_shape =(784,)))
# # # model = Sequential()
# model.add(Dense(units=10, input_shape=(28 * 28,)))   
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(80, activation= 'relu'))
# model.add(Dense(80, activation= 'relu'))
# model.add(Dense(80, activation= 'relu'))
# model.add(Dense(10, activation= 'softmax'))
# model.summary()
