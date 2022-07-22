# 증폭해서 numpy 저장
from colorsys import yiq_to_rgb
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D ,Flatten, Dense, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

x_train = np.load('d:/study_data/_save/_npy/keras49_1_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_1_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_1_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_1_test_y.npy')

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5),   
                 padding='same',
                 input_shape=(28, 28, 1)))                                 
model.add(MaxPooling2D())                                               
model.add(Conv2D(32, (2,2), activation= 'relu'))                                                            
model.add(Flatten())                        
model.add(Dense(16, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일 구성 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
               restore_best_weights = True)     
        
hist = model.fit_generator(x_train, epochs=100, 
                    steps_per_epoch=32,  # steps_per_epoch=32 데이터를 batch size로 나눈것. 160/5 =32 
                    validation_data=x_train,
                    validation_steps=4)
                    

#4. 평가, 예측\
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])
############################################

y_predict = model.predict(x_test)
print('predict : ',y_predict[-1])
print('y_predict: ',y_predict[-1])

# y_predict = tf.argmax(y_predict,axis=1) 
# print(y_predict)
# y_test = tf.argmax(y_test,axis=1)         # argmax 형태가 맞지만, 값이 너무 달라 비교가 안될때 사용.
                                        #   [6,7,9,10]   >> 3 반환. (0123 순서로 계산.)
                                        #   [3,8,1,2]    >> 1 반환. 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

# loss :  0.6270356774330139
# accuracy :  0.8529999852180481
# predict :  [2.4800003e-02 7.0197399e-05 3.6593714e-01 2.1013040e-02 3.1523156e-01
#  9.6060947e-04 2.7059782e-01 5.9307378e-04 1.6371348e-04 6.3286209e-04]