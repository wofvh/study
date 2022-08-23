import tensorflow as tf
tf.compat.v1.set_random_seed(123)

aa = tf.compat.v1.Variable(tf.random_normal([1]),name='weight')

print(aa)


#1.초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(aa)
print('aaa',aaa)        # aaa [-1.5080816]
sess.close()


#2. 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb= aa.eval(session =sess)         # 이과정을 거처야 변수로 변환.
print('bbb',bbb)
sess.close()

#2. 초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = aa.eval()
print('ccc',ccc)
sess.close()





