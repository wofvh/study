import tensorflow as tf

tf.set_random_seed(123)

#1. 데이터 
x = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(1234,dtype=tf.float32)
b = tf.Variable(72,dtype=tf.float32)

#2. 모델
hypothesis = x * w + b          # y= wx+b


# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))            # mse   # square 제곱  (h-y)제곱 / n
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)               # 경사경법
train = optimizer.minimize(loss)                           
# 가장 낮은값을 찾는다. model.compile(loss='mse',optimizer='sgd')
 
 # 3-2 훈련
with tf.compat.v1.Session() as sess :                       # with  함께 쓰겠다. with문 안에서 작업을 하면 close로 마무리 하지않아도 된다.
                                                            # with를 사용하지 않으면 sess.close()를 사용해서 닫아줘야한다. 
# sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())  # 초기화
    # 변수형은 그래프를 실행하기 전에 초기화해줘야 변수에 지정이 가능하다.
    
    
    epochs = 2001

    for step in range(epochs):
        sess.run(train)                                         # model.fit()
        if step %50 == 0:                                       # 50의 배수일 때 
            print(step,sess.run(loss),sess.run(w),sess.run(b))
        
# sess.close() 




