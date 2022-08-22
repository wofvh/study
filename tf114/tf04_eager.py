import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())       # False

# 즉시실행모드!
tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly())       # False

hello = tf.constant('helloworld')

sess = tf.compat.v1.Session()
print(sess.run(hello))
