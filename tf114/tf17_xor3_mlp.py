import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error

#1.데이터

x_data = [[0,0],[0,1],[1,0],[1,1]]  # (4,2)
y_data = [[0],[0],[1],[1]]          # (4,)


#2. 모델
#input layer
x= tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y= tf.compat.v1.placeholder(tf.float32,shape=[None,1])

#hidden layer
w1 =tf.compat.v1.Variable(tf.random_normal([2, 20]))
b1= tf.compat.v1.Variable(tf.random_normal([20]))

h1 = tf.matmul(x,w1)+b1

w2 =tf.compat.v1.Variable(tf.random_normal([20, 30]))
b2= tf.compat.v1.Variable(tf.random_normal([30]))

h2 =tf.sigmoid(tf.matmul(h1,w2)+b2)

w3 =tf.compat.v1.Variable(tf.random_normal([30, 20]))
b3= tf.compat.v1.Variable(tf.random_normal([20]))

h3 =tf.sigmoid(tf.matmul(h2,w3)+b3)

#output layer
w4 = tf.compat.v1.Variable(tf.random_normal([20, 1]))
b4= tf.compat.v1.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(h3,w4)+b4)


# loss = binary_cross_entropy_with_logits # sigmoid
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w1) + b1)
# hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w2) + b2)

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

# 컴파일
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.16)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    # _, loss_val, w_val1, w_val2, w_val3 = sess.run([train, loss, w1, w2, w3], 
    #                                                feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,y:y_data})
    # print(epochs, '\t', 'loss:',loss_val, '\t', '국어',w_val1, '\t', '영어',w_val2, '\t', '수학',w_val3)
    
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x:x_data,y:y_data})
    if epochs %500 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', h_val)


#4. 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_data})
y_predict = sess.run(tf.cast(y_predict > 0.5, dtype=tf.float32))

# print(y_test.shape,y_predict.shape) # (179, 1) (712, 1)

acc_score = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc_score)

# mse = mean_absolute_error(y, h_val)
# print('mse : ', mse)

sess.close()

# 훈련



