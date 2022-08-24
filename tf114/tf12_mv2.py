import tensorflow as tf 
tf.compat.v1.set_random_seed(72)

x_data = [[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79],[17,66,79],[99,33,100],[17,66,79],[17,66,79]]
y_data = [[152],[185],[180],[205],[142],[142],[205],[142],[142]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])    # 열
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])    # 행

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name='weight')  # 
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='bias')  

hypothesis = tf.compat.v1.matmul(x,w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    # _, loss_val, w_val1, w_val2, w_val3 = sess.run([train, loss, w1, w2, w3], 
    #                                                feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,y:y_data})
    # print(epochs, '\t', 'loss:',loss_val, '\t', '국어',w_val1, '\t', '영어',w_val2, '\t', '수학',w_val3)
    
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x:x_data,y:y_data})
    print(epochs, '\t', 'loss:',loss_val, '\t', h_val)
    
#4. 예측
# predict =  x1*w_val1 + x2*w_val2 + x3*w_val3 + b   # predict = model.predict

# y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
# print("[152., 185., 180., 196., 142.]예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
r2 = r2_score(y_data, h_val)
print('r2 : ', r2)

mae = mean_absolute_error(y_data, h_val)
print('mae : ', mae)


