import imp
import tensorflow as tf
import keras
import numpy as np


tf.compat.v1.set_random_seed(123)
tf.compat.v1.disable_eager_execution()

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) =mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.


#2. 모델 
x = tf.compat.v1.placeholder(tf.float32,[None,28,28,1])     # input_shape
y = tf.compat.v1.placeholder(tf.float32,[None,10])          # output_shape

# layer1
w = tf.compat.v1.get_variable('w1',shape=[2, 2, 1, 128])     # 1 = 그다음 input과 맞춰야함. 
                    # 2,2 = 커널사이즈/ 1 = 칼라 / 64(filter) outputload 
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([10]))
                    
L = tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='SAME') 
L = tf.nn.relu(L)
L_maxpool = tf.nn.max_pool2d(L,ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")

# model.add(Conv2d(64,kernal_size-(2,2),input_shape=(28,28,1),activation='relu')                     
# stride=[1, 1, 1, 1] 이동 가운데 두개가 사이즈를 정하고 양쪽두개는 쉐입을 맞춰주기위함.
print(w)        # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L)        # Tensor("Conv2D:0", shape=(?, 28, 28, 128), dtype=float32)
print(L_maxpool)

# layer2
w2 = tf.compat.v1.get_variable('w2',shape=[3, 3, 128, 64])     # 1 = 그다음 input과 맞춰야함. 
L2 = tf.nn.conv2d(L_maxpool,w2,strides=[1,1,1,1], padding='VALID') 
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool2d(L2,ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
print(w2)
print(L2)           # Tensor("Selu:0", shape=(?, 12, 12, 64), dtype=float32)
print(L2_maxpool)   # Tensor("MaxPool_1:0", shape=(?, 6, 6, 64), dtype=float32)


# layer3
w3 = tf.compat.v1.get_variable('w3',shape=[3, 3, 64, 32])     # 1 = 그다음 input과 맞춰야함. 
L3 = tf.nn.conv2d(L2_maxpool,w3,strides=[1,1,1,1], padding='VALID') 
L3 = tf.nn.relu(L3)
# L3_maxpool = tf.nn.max_pool_v2(L,ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
            
print(L3)       # Tensor("Elu:0", shape=(?, 5, 5, 128), dtype=float32)

# FLATTEN(layer3) 
L_flat = tf.reshape(L3,[-1,4*4*32])
print('platten:',L_flat)        # platten: Tensor("Reshape:0", shape=(?, 512), dtype=float32)

# layer4 DNN
w4 = tf.compat.v1.get_variable('w4',shape=[4*4*32,100],)
                    #  initializer=tf.compat.v1.contrib.layers.xavier_initializer())   # 프로젝트
b4 = tf.Variable(tf.compat.v1.random_normal([100]),name='b5')
L4 = tf.nn.selu(tf.matmul(L_flat,w4) +b4)
L4 = tf.nn.dropout(L4, rate=0.3)   # rate=0.3

# layer5 DNN
w5 = tf.compat.v1.get_variable('w5',shape=[100,10],)
                    #  initializer=tf.contrib.layers.xavier_initializer())   # 프로젝트
b5 = tf.Variable(tf.compat.v1.random_normal([10]),name='b5')
L5 = tf.matmul(L4,w5) +b5
h = tf.nn.softmax(L5)
print(h)            # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

# loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(h)),axis=1)
loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=h)  )

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

t_epochs = 30
batch_size = 100
total_batch = int(len(x_train)/batch_size)

for epochs in range(t_epochs):
    avg_loss = 0
    for i in range(total_batch):
        start = i * batch_size
        end = start +batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict ={x:batch_x,y:batch_y}
        batch_loss, _ = sess.run([loss,optimizer],feed_dict=feed_dict)        
        
        avg_loss +=batch_loss /total_batch
        
    print('Epoch : ',  '%04d' %(epochs + 1), 'loss : {:.9f}'.format(avg_loss))  
    
prediction = tf.compat.v1.equal(tf.argmax(h,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(prediction, tf.float32))
print( 'acc:',sess.run(acc, feed_dict={x:x_test,y:y_test}))
# verbose에 acc 넣기 
'''
    _, loss_val, h_val = sess.run([train, loss, h], 
                                                   feed_dict={x:x_train,y:y_train})
    if epochs %5 ==0:
        print(epochs, '\t', 'loss:',loss_val, '\t', h_val)

#4. 예측
# y_predict =sess.run(tf.argmax(h_val))
# y_test = sess.run(tf.argmax(y_test))

# y_predict = sess.run(tf.cast(h_val>=0.5, dtype=tf.float32))   # 참이면 1 , 거짓이면 0
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score ,mean_squared_error

y_predict = sess.run(h, feed_dict={x:x_test})
# y_predict = sess.run(tf.cast(y_predict > 0.5, dtype=tf.float32))

print(y_test.shape,y_predict.shape) # (179, 1) (712, 1)
y_predict = sess.run(tf.argmax(y_predict, axis=1))
y_test = sess.run(tf.argmax(y_test, axis=1))

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)

# mse = mean_absolute_error(y, h_val)
# print('mse : ', mse)

sess.close()




'''



