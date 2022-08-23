from re import I
import tensorflow as tf
sess = tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()  # 즉시실행모드 종료.

x = tf.Variable([2],dtype=tf.float32)
y = tf.Variable([3],dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()       # 초기화실행
sess.run(init)                                           # 반드시 실행해야한다.

print(sess.run(x+y))

#  변수형은 그래프를 실행하기 전에 초기화해줘야 변수에 지정이 가능하다.

