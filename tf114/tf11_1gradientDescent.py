import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1,2,3]
y_train = [1,2,3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

tf.set_random_seed(72)       

# w = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),name='weight')


hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis-y))

lr = 0.1
gradient = tf.reduce_mean((w * x - y ) * x)
descent = w - lr * gradient
update = w.assign(descent)

w_histroy = []
loss_histroy = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _,loss_v,w_v =sess.run([update,loss,w],feed_dict={x:x_train,y:y_train})
    print(step,'\t',loss_v, '\t',w_v )
    w_histroy.append(w_v)
    loss_histroy.append(loss_v)
          
sess.close()

print('-=======w history=====')
print(w_histroy) 
print('========loss_history========')
print(loss_histroy)
  
        
plt.plot(w_histroy,loss_histroy)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()      
        