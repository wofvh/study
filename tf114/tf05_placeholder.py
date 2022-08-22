#데이터 정의 
# 선언과 동시에 초기화 하는 것이 아니라 일단 선언 후 그 다음 값을 전달한다. 따라서 반드시 실행 시 데이터가 제공되어야 한다. 
# 여기서 값을 전달한다고 되어 있는데 이는 데이터를 상수값을 전달함과 같이 할당하는 것이 아니라 다른 텐서(Tensor)를 placeholder에 
# 맵핑 시키는 것이라고 보면 된다.


import numpy as np
import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())       # True

tf.compat.v1.disable_eager_execution()  # 즉시실행모드 종료.

print(tf.executing_eagerly())       # False

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node2 = tf.add(node1,node2)

sess = tf.compat.v1.Session()


a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

add_nod =a + b 
print(sess.run(add_nod,feed_dict={a:3, b:4.5}))             # 7.5
print(sess.run(add_nod,feed_dict={a:[1,3], b:[2,4]}))       # [3. 7.]
# placeholder로 공간을 만들고 fee_dict에 값을 넣는다.


add_and_triple = add_nod * 3 
print(sess.run(add_and_triple,feed_dict={a:3, b:4.5}))

exit()
print(sess.run(add_nod,feed_dict={a:3, b:4.5}))             # 7.5
print(sess.run(add_nod,feed_dict={a:[1,3], b:[2,4]}))       # [3. 7.]





