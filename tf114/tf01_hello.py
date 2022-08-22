import tensorflow as tf
print(tf.__version__)


# print('hello world')
hello = tf.constant('hello world') # constant 상수(고정값)
# print(hello)

# sess =tf.Session()
sess =tf.compat.v1.Session()

print(sess.run(hello))             # tensorflow1은 프린트할 때 반드시 session을 거쳐야한다. 








