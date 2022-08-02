
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

train_datagen = ImageDataGenerator(             # 이미지를 수치화. 증폭도 가능. 
    rescale=1./255,                             # 다른 처리 전에 데이터를 곱할 값/ 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 
                                              
    horizontal_flip=True,                       # 이미지의 절반을 수평으로 무작위로 뒤집기 위한 것.
    vertical_flip=True,                         # 이미지의 절반을 수직으로 무작위로 뒤집기 위한 것.
    width_shift_range=0.1,                      # width_shift그림을 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    height_shift_range=-0.1,                    # height_shift 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    rotation_range=5,                           # 사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
    zoom_range=1.2,                             # 내부 사진을 무작위로 확대하기 위한 것입니다
    shear_range=0.7,                            # 무작위로 전단 변환 을 적용하기 위한 것입니다. # 찌그러,기울려 
    fill_mode='nearest'                         # 회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.
)

test_datagen =ImageDataGenerator(               # 평가데이터는 증폭하지 않는다. (수정x)
    rescale=1./255                              # 가장 작은값을 0~255 가장큰값으로 가장 작은값을 빼고 나눈다. 0~1 사이
)

xy = train_datagen.flow_from_directory(
    'D:\study_data\_data\season\dataset',
    target_size=(150,150),                       
    batch_size=4000,
    class_mode='categorical',                        
    shuffle=True,
    )                                            

x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=58525 )
                                                  
augument_size = 500                  
randidx =np.random.randint(x_train.shape[0],size=augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

x_augumented = train_datagen.flow(x_augumented, y_augumented, batch_size=augument_size, shuffle=False).next()[0]

# 원본train + 증폭train 
x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

np.save('d:/study_data/_save/_npy/project_train11_x.npy', arr =x_train)
np.save('d:/study_data/_save/_npy/project_train11_y.npy', arr =y_train)
np.save('d:/study_data/_save/_npy/project_test11_x.npy', arr =x_test)
np.save('d:/study_data/_save/_npy/project_test11_y.npy', arr =y_test)


