from sklearn import datasets
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype, load_digits
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# tf.set_random_seed(66)

#1. 데이터
datasets = load_digits()
x_data = datasets.data
y_data = datasets.target

from tensorflow.keras.utils import to_categorical
y_data = to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.9, shuffle=True, random_state=123,stratify=y_data)

# y_train =np.array(y_train, dtype='float32')

print(x_data.shape[1]) #64
print(y_data.shape[1]) #10



#2. 모델구성 // 시작 
x = tf.compat.v1.placeholder(tf.float32,shape=[None, x_data.shape[1]])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1],y_data.shape[1]]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,y_data.shape[1]]))

y = tf.compat.v1.placeholder(tf.float32,shape=[None, y_data.shape[1]])

###############################################################
# w1 =tf.compat.v1.Variable(tf.random_normal([2, 20]))
# b1= tf.compat.v1.Variable(tf.random_normal([20]))

h1 = tf.matmul(x,w)+b

w2 =tf.compat.v1.Variable(tf.random_normal([y_data.shape[1], 30]))
b2= tf.compat.v1.Variable(tf.zeros([30]))

h2 =tf.nn.sigmoid(tf.matmul(h1,w2)+b2)

w3 =tf.compat.v1.Variable(tf.random_normal([30, 20]))
b3= tf.compat.v1.Variable(tf.zeros([20]))

h3 =tf.nn.sigmoid(tf.matmul(h2,w3)+b3)

#output layer
w4 = tf.compat.v1.Variable(tf.random_normal([20, y_data.shape[1]]))
b4= tf.compat.v1.Variable(tf.zeros([y_data.shape[1]]))

hypothesis = tf.nn.softmax(tf.matmul(h3,w4) +b4)

#############################################

# loss = tf.reduce_mean(tf.reduce_sum(y*tf.log(hypothesis),axis=1))  
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=tf.matmul(x,w) +b)  

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)

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

# accuracy_score :  0.9555555555555556

