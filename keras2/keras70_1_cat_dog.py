# https://www.kaggle.com/c/dogs-vs-vats/data
from matplotlib.pyplot import hist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

# 1 정상 0 문제 

#1. 데이터

train_dataen = ImageDataGenerator(              # 이미지를 수치화. 증폭도 가능. 
    rescale=1./255,                             # 다른 처리 전에 데이터를 곱할 값입니다. 원본 이미지는 0-255의 RGB 계수로 구성되지만 이러한 값은 모델이 처리하기에는 너무 높기 때문에(주어진 일반적인 학습률) 
                                                # 1/255로 스케일링하여 대신 0과 1 사이의 값을 목표로 합니다.
    # horizontal_flip=True,                       # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수평 비대칭에 대한 가정이 없을 때 관련이 있습니다
    # vertical_flip=True,                         # 이미지의 절반을 가로로 무작위로 뒤집기 위한 것입니다. 수직 비대칭에 대한 가정이 없을 때 관련이 있습니다
    # width_shift_range=0.1,                      # width_shift그림을 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    # height_shift_range=-0.1,                    # height_shift 수직 또는 수평으로 무작위로 변환하는 범위(총 너비 또는 높이의 일부)입니다.
    # rotation_range=5,                           # 사진을 무작위로 회전할 범위인 도(0-180) 값입니다.
    # zoom_range=1.2,                             # 내부 사진을 무작위로 확대하기 위한 것입니다
    # shear_range=0.7,                            # 무작위로 전단 변환 을 적용하기 위한 것입니다. # 찌그러,기울려 
    # fill_mode='nearest'                         # 회전 또는 너비/높이 이동 후에 나타날 수 있는 새로 생성된 픽셀을 채우는 데 사용되는 전략입니다.
)

test_datagen =ImageDataGenerator(               # 평가데이터는 증폭하지 않는다. (수정x)
    rescale=1./255
)

xy_train = train_dataen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/training_set/training_set/',
    target_size=(100,100),                       # 사진을 가져올때 사이즈 조정. 
    batch_size=5,
    class_mode='binary',                         # 흑백이라 binary, 칼라 catagorical
    shuffle=True,
                         # color_mode 디폴트 칼라. 
    )                                            # Found 160 images belonging to 2 classes. > 160개 사진과 0,1 2 class 생성. 

xy_test = train_dataen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test_set/test_set/',
    target_size=(100,100),                       # 사진을 가져올때 사이즈 조정. 
    batch_size=5,
    class_mode='binary',                         # 흑백이라 binary, 칼라 catagorical
    shuffle=True,
           
    )                                            # Found 120 images belonging to 2 classes. > 160개 사진과 0,1 2 class 생성. 

print(xy_train)                          

print(xy_train[0][0].shape, xy_train[0][1].shape) # (500, 150, 150, 3) (500,)
print(xy_test[0][0].shape, xy_test[0][1].shape)   # (500, 150, 150, 3) (500,)                                      

np.save('d:/study_data/_save/_npy/keras47_1_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras47_1_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras47_1_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras47_1_test_y.npy', arr=xy_test[0][1])

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_train[0][0]
y_test = xy_train[0][1]


#2 모델구성 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D

from keras.applications import VGG16
models = VGG16(weights='imagenet',include_top=False)

# models.trainable=False
# models.summary()

model =Sequential()
model.add(models)
model.add(Flatten())
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
    

#3. 컴파일.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1])          # 배치를 최대로 잡으면 가능
hist = model.fit_generator(xy_train, epochs=100, 
                    steps_per_epoch=32,  # steps_per_epoch=32 데이터를 batch size로 나눈것. 160/5 =32 
                    validation_data=xy_test,
                    validation_steps=4)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-1])
print('val_loss : ',val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('predict : ',y_predict[-1])


# loss :  0.670547604560852
# val_loss :  0.7142734527587891
# accuracy :  0.543749988079071
# val_accuracy :  0.5


# VGG16
# loss :  0.6915820837020874
# val_loss :  0.6841166615486145
# accuracy :  0.543749988079071
# val_accuracy :  0.699999988079071
# 1/1 [==============================] - 0s 27ms/step - loss: 0.6888 - accuracy: 0.6000
# predict :  [0.5116241]