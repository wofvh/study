import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]
             ,[9,8,7,6,5,4,3,2,1,0]])   # -> y= w1x1 + w2x2 +b
y = np.array([11,12,13,14,15,16,17,18,19,20])    # (10,)
print (x.shape)  #(3,10)    # x = 행3, 열,10 
print (y.shape)  #(10,)

# x = x.transpose()                     # transpose 행과열의 위치변경 방법1 
x = x.T                                 # transpose 행과열의 위치변경 방법2

print(x.shape)   #(10, 3) (10,)



#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=3))    # (100,3) > (None, 3) 
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 10)                40
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 55
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 18
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 117
# Trainable params: 117
# Non-trainable params: 0





