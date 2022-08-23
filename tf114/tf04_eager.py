import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())       # True

# 즉시실행모드! (2.0 이상버전)
tf.compat.v1.disable_eager_execution()  # 즉시실행모드를 사용하겠다. 
# 1.0 버전은 즉시실행모드가 안되기 때문에 위에 코드를 사용하지 않아도 된다. 
# 2.0 버전 이상에서는 즉시실행모드가 자동으로 있기 때문에 즉시실행모드를 사용하지 않겠다고 해야한다. 

print(tf.executing_eagerly())       # False

hello = tf.constant('helloworld')

sess = tf.compat.v1.Session()
print(sess.run(hello))
