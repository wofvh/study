import tensorflow as tf

node1 = tf.constant(20000.0)
node2 = tf.constant(31351351.0)

#덧셈 node3
#뺄셈 node4
#곱셈 node5
#나눗셈 node6

sess = tf.Session()

node3 = node1 + node2
node3_ = tf.add(node1,node2)

node4 = node1 - node2
node4_ = tf.subtract(node1,node2)

node5 = node1 * node2
node5_ = tf.multiply(node1,node2)

node6 = node1 / node2
node6_ = tf.divide(node1,node2)


print('node3더하기',sess.run([node3, node3_]))
print('node4빼기',sess.run([node4, node4_]))
print('node5곱하기',sess.run([node5, node5_]))
print('node6나누기',sess.run([node6, node6_]))
