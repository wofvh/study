import tensorflow as tf
tf.compat.v1.set_random_seed(123)

#1. 데이터
         # 첫번 두번 세번 네번 다섯번
x1_data = [73., 93., 89., 96., 73.]       # 국어                  .을 찍은 이유는 float형태로 나타내주기위해서
x2_data = [80., 88., 91., 98., 66.]       # 영어
x3_data = [75., 93., 90., 100., 70.]      # 수학
y_data = [152., 185., 180., 196., 142.]   # 환산점수

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight1')
w2 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight2')
w3 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight3')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000016)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    # _, loss_val, w_val1, w_val2, w_val3 = sess.run([train, loss, w1, w2, w3], 
    #                                                feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,y:y_data})
    # print(epochs, '\t', 'loss:',loss_val, '\t', '국어',w_val1, '\t', '영어',w_val2, '\t', '수학',w_val3)
    
    _, loss_val, h_val = sess.run([train, loss, hypothesis], 
                                                   feed_dict={x1:x1_data, x2:x2_data, x3:x3_data,y:y_data})
    print(epochs, '\t', 'loss:',loss_val, '\t', h_val)
    
#4. 예측
# predict =  x1*w_val1 + x2*w_val2 + x3*w_val3 + b   # predict = model.predict

# y_predict = sess.run(predict, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
# print("[152., 185., 180., 196., 142.]예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, h_val)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, h_val)
print('mae : ', mae)

# [152., 185., 180., 196., 142.]예측 :  
#     [151.76665 184.52641 180.95152 195.52644 142.31248]
# r2스코어 :  0.9992842032673048
# mae :  0.4889007568359375

