import tensorflow as tf

tf.set_random_seed(123)       

# 1. 데이터 
# x = [1,2,3,4,5]
# y = [1,2,3,4,5]
x_train = tf.placeholder(tf.float32,shape=[None,5])  # shape=[None]  1차원일 때만 자동으로 쉐입을 잡아준다. 
y_train = tf.placeholder(tf.float32,shape=[2,5])

# w = tf.Variable(111,dtype=tf.float32)
# b = tf.Variable(72,dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)    
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)

# 2. 모델
hypothesis = x_train * w + b          

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))            
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)              
train = optimizer.minimize(loss)                           
 
 # 3-2 훈련
with tf.compat.v1.Session() as sess :                                                                 
    sess.run(tf.global_variables_initializer())  
   
    epochs = 2001

    for step in range(epochs):
        # sess.run(train)                                        
        _, loss_val, w_val, b_val = sess.run([train, loss,w,b],     
                                    feed_dict = {x_train:[[1,2,3,4,5],[1,2,3,4,5]], y_train:[[1,2,3,4,5],[1,2,3,4,5]]})
        if step %50 == 0:                                        
            print(step, loss_val, w_val, b_val)


####################################################################

with tf.compat.v1.Session() as sess :                                                                 

    sess.run(tf.global_variables_initializer()) 
    x_data = [[1,2,3,4,5],[1,2,3,4,5]]            
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,5])
    y_predict = x_test * w_val + b_val              # y_predict = model.predict(x_test)
    print('[6,7,8,]예측',sess.run(y_predict,feed_dict={x_test:x_data}))

####################################################################

# sess = tf.compat.v1.Seesion()
# sess.run(tf.global_variables_initializer()) 
# x_data = [6,7,8]            
# x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])
# y_predict = x_test * w_val + b_val              # y_predict = model.predict(x_test)
# print('[6,7,8,]예측',sess.run(y_predict,feed_dict={x_test:x_data}))

        





