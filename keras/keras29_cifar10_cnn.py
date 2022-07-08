#칼라
#분류 
# 32
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from tensorflow.keras.datasets import mnist, cifar10
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) =cifar10.load_data()

print(x_train.shape, y_train.shape)    
print(x_test.shape, y_test.shape)      

x_train = x_train.reshape(60000, 28, 28,1)  # input 28,28,1 
x_test = x_test.reshape(10000, 28, 28,1)    # 

print(x_train.shape)
print(np.unique(y_train, return_counts =True))