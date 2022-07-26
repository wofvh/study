
from colorsys import yiq_to_rgb

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D ,Flatten, Dense, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar100 , fashion_mnist

train_datagen = ImageDataGenerator(              # 이미지를 수치화. 증폭도 가능. 
    rescale=1./255,                             # 다른 처리 전에 데이터를 곱할 값입니다. 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 
                                                # 1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다.
    horizontal_flip=True,                       # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다
    vertical_flip=True,                         # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수직 비대칭에 대한 가정이 없을 때 관련이 있습니다
    width_shift_range=0.1,                      # width_shift그림을 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    height_shift_range=-0.1,                    # height_shift 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    rotation_range=5,                           # 사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
    zoom_range=1.2,                             # 내부 사진을 무작위로 확대하기 위한 것입니다
    shear_range=0.7,                            # 무작위로 전단 변환 을 적용하기 위한 것입니다. # 찌그러,기울려 
    fill_mode='nearest'                         # 회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.
)

test_datagen =ImageDataGenerator(               # 평가데이터는 증폭하지 않는다. (수정x)
    rescale=1./255
)

xy = train_datagen.flow_from_directory(
    'D:\study_data\_data\season\dataset',
    target_size=(150,150),                       
    batch_size=2000,
    class_mode='binary',                        
    shuffle=True,
  
    )                                            

x = xy[0][0]
y = xy[0][1]

print(x.shape,y.shape)  # (5, 100, 100, 3) (5,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.725,
                                                    shuffle=True,
                                                    random_state=58525
                                                    )
                          

print(x_train.shape, x_train.shape) #  (1450, 150, 150, 3) (1450, 150, 150, 3)
print(y_test.shape, y_test.shape)   # (550,) (550,)                          


augument_size = 2000                     # 반복횟수
randidx =np.random.randint(x_train.shape[0],size=augument_size)

print(np.min(randidx),np.max(randidx))      # random 함수 적용가능. 
print(type(randidx))            # <class 'numpy.ndarray'>  

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

print(x_augumented.shape)       # (40000, 150, 150, 1)
print(y_augumented.shape)       # (40000,)

# x_train = x_train.reshape(50000,32,32,3)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2],
                                    3)

xy_train = train_datagen.flow(x_train,y_train,
                                batch_size = augument_size,
                                shuffle=False)

x_train1 =np.concatenate((x_train,x_augumented))
y_train2 =np.concatenate((y_train,y_augumented))

xy_augumented = test_datagen.flow(x_train1, y_train2,
                                batch_size = augument_size,
                                shuffle=False)

np.save('d:/study_data/_save/_npy/personalproject_train_x.npy', arr=xy_augumented[0][0])
np.save('d:/study_data/_save/_npy/personalproject_train_y.npy', arr=xy_augumented[0][1])
np.save('d:/study_data/_save/_npy/personalproject_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/personalproject_test_y.npy', arr=y_test)

