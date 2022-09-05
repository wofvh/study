import imp
import tensorflow as tf
import keras
import numpy as np


tf.compat.v1.set_random_seed(123)


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
w = tf.compat.v1.get_variable('w1',shape=[2, 2, 1, 64])     # 1 = 그다음 input과 맞춰야함. 
                    # 2,2 = 커널사이즈/ 1 = 칼라 / 64(filter) outputload 
L = tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='VALID') 
# model.add(Conv2d(64,kernal_size-(2,2),input_shape=(28,28,1))                     
# stride=[1, 1, 1, 1] 이동 가운데 두개가 사이즈를 정하고 양쪽두개는 쉐입을 맞춰주기위함.


print(w)        # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L)        # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)

            





