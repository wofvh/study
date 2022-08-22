import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2],dtype=tf.float32)
y = tf.Variable([3],dtype=tf.float32)

print(sess.run(x+y))

