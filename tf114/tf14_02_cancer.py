from sklearn import datasets
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.set_random_seed(123)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)    # (569, 30) (569,)
y_data = y_data.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size=0.8, shuffle=True, random_state=123,stratify=y_data)

y_train =np.array(y_train, dtype='float32')

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None,1])
w = tf.compat.v1.Variable(tf.zeros([30,1]), name='weight')    # y = x * w  
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias') 

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y_train*tf.log(hypothesis)+(1-y_train)*tf.log(1-hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(201):
  
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x:x_train,y:x_test})
    if epochs %5 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', h_val)

#4. 예측

y_predict = sess.run(tf.cast(h_val>=0.5, dtype=tf.float32))   # 참이면 1 , 거짓이면 0
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error
sess.close()
print( y_data.shape,y_predict.shape)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

mse = mean_absolute_error(y_data, h_val)
print('mse : ', mse)



