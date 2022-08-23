from pickletools import optimize
import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터 
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]

w = tf.Variable(10,dtype=tf.float32)
b = tf.Variable(11,dtype=tf.float32)


#2. 모델
hypothesis = x * w + b          # y= wx+b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))            # mse   # square 제곱  (h-y)제곱 / n
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)               # 경사경법
train = optimizer.minimize(loss)                           
# 가장 낮은값을 찾는다. model.compile(loss='mse',optimizer='sgd')
 
 # 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
 
for step in range(2001):
    sess.run(train)                                         # model.fit()
    if step %20 == 0:
        print(step,sess.run(loss),sess.run(w),sess.run(b))
        
sess.close() 
