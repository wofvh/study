from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from tensorflow.keras.datasets import mnist
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) =mnist.load_data()

print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28,1)  # input 28,28,1 
x_test = x_test.reshape(10000, 28, 28,1)    # 

print(x_train.shape)
print(np.unique(y_train, return_counts =True))

# 실습 acc 0.98이상 
# 원핫인코딩 

model = Sequential()
#model.add(Dense(units=10, input_shape=(3,)))     # batch_size(행),input_dim(열))           # input_shape=(10, 10, 3)
#    (input_dim + bias) * units = summary Param 개수(Dense모델)

model.add(Conv2D(filters=64, kernel_size=(3, 3),   # 출력(4,4,10)                                       # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(28, 28, 1)))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라 
                                                                                           # kernel_size(2*2) * 바이어스(3) + 10(output)
model.add(MaxPooling2D())

 #    (kernel_size * channls) * filters = summary Param 개수(CNN모델)  
model.add(Conv2D(32, (2,2), 
                 padding = 'valid',         # 디폴트값(안준것과 같다.) 
                 activation= 'relu'))    # 출력(3,3,7)                                                     
model.add(Flatten()) # (N, 63)
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()



# (kernel_size * channls + bias) * filters(output) = summary Param 개수

# tf.keras.layers.Dense(
#     units,                                # output 로드 개수 10 
#     activation=None,
#     use_bias=True,                        # 
#     kernel_initializer="glorot_uniform",  # 레이어 초기화
#     bias_initializer="zeros",             # 레이어 초기화
#     kernel_regularizer=None,              # 정규화, 규제화 
#     bias_regularizer=None,                # 정규화, 규제화 
#     activity_regularizer=None,            # 정규화, 규제화 
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs)

#      model.add(Dense(10, activation ='relu', input_dim =8)
#      2차원일때 input shape ) Dense > (batch_size(행),input_dim(열))

