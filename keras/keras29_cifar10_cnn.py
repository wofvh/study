#칼라
#분류 
# 32
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from keras.datasets import mnist, cifar10 , fashion_mnist
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) =cifar10.load_data()

print(x_train.shape, y_train.shape)         # (50000, 32, 32, 3)
print(x_test.shape, y_test.shape)           # (10000, 32, 32, 3)

x_train = x_train.reshape   #(50000, 32, 32, 3)  # input 32,32,1 
x_test = x_test.reshape     #(10000, 32, 32, 3)    # 

print(x_train.shape)
print(np.unique(y_train, return_counts =True))


