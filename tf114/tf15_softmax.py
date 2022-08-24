import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(123)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]     # (8, 4)
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]       # (8, 3)

#2. 모델구성 // 시작 
x = tf.compat.v1.placeholder(tf.float32,shape=[None, 4])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]))

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,3]))

y = tf.compat.v1.placeholder(tf.float32,shape=[None, 3])

hypothesis = tf.nn.softmax(tf.matmul(x,w) +b)

loss = -tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))   
#loss = 'categorical_crossentropy'

# optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = tf.train.AdamOptimizer(learning_rate= 0.006).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(200):
  
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x:x_data,y:y_data})
    if epochs %50 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', h_val)

#4. 예측
y_predict =sess.run(tf.argmax(h_val))
y_data = sess.run(tf.argmax(y_data))

# y_predict = sess.run(tf.cast(h_val>=0.5, dtype=tf.float32))   # 참이면 1 , 거짓이면 0
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error


print(y_data.shape,y_predict.shape) # (179, 1) (712, 1)

acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)

# mse = mean_absolute_error(y, h_val)
# print('mse : ', mse)

sess.close()






