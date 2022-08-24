import tensorflow as tf
tf.compat.v1.set_random_seed(123)


#1. 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]  # (6,2)
y_data = [[0],[0],[0],[1],[1],[1]]              # (6,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])    # 열
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])    # 행

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')  # 
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias')  


hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)
# model.add(Dense(1,activation='sigmoid',input_dim=2))

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary_crosstropy model.compile(loss='binary_crossentropy')

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
y_predict = sess.run(tf.cast(h_val>0.5, dtype=tf.float32))   # 참이면 1 , 거짓이면 0


# y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
# print("[152., 185., 180., 196., 142.]예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)

mse = mean_absolute_error(y_data, h_val)
print('mse : ', mse)

