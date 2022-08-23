#실습
#lr수정 epoch를 100번 이하로 줄이고, step = 100 이하, w =1.99, b = 0.99

x_train_data = [1,2,3]
y_train_data = [3,5,7]

import tensorflow as tf

tf.set_random_seed(123)       

# 1. 데이터 
# x = [1,2,3,4,5]
# y = [1,2,3,4,5]
x_train = tf.placeholder(tf.float32,shape=[None])  # shape=[None]  1차원일 때만 자동으로 쉐입을 잡아준다. 
y_train = tf.placeholder(tf.float32,shape=[None])

# w = tf.Variable(111,dtype=tf.float32)
# b = tf.Variable(72,dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)    
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)

# 2. 모델
hypothesis = x_train * w + b          

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))            
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.16)              
train = optimizer.minimize(loss)                           
 
 # 3-2 훈련
with tf.compat.v1.Session() as sess :                                                                 
    sess.run(tf.global_variables_initializer())  
   
    epochs = 100

    for step in range(epochs):
        # sess.run(train)                                        
        _, loss_val, w_val, b_val = sess.run([train, loss,w,b],     
                                    feed_dict = {x_train:x_train_data, y_train:y_train_data})
        if step %50 == 0:                                        
            print(step, loss_val, w_val, b_val)
   
    x_test_data = [6,7,8]            
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])
    y_predict = x_test * w_val + b_val              # y_predict = model.predict(x_test)
    print('[6,7,8,]예측',sess.run(y_predict,feed_dict={x_test:x_test_data}))
          
####################################################################

# with tf.compat.v1.Session() as sess :                                                                 

#     sess.run(tf.global_variables_initializer()) 
#     x_test_data = [[1,2,3,4,5],[1,2,3,4,5]]            
#     x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,5])
#     y_predict = x_test * w_val + b_val              # y_predict = model.predict(x_test)
#     print('[6,7,8,]예측',sess.run(y_predict,feed_dict={x_test:x_test_data}))

####################################################################



        





