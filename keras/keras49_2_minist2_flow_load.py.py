# 넘파이에서 불러와서 모델구성
# 성능비교
# 넘파이에서 불러와서 모델구성
# 성능비교
# from click import argument
# from sklearn.utils import shuffle
from colorsys import yiq_to_rgb
from tensorflow.keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D ,Flatten, Dense, Dropout, Input
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

x_train = np.load('d:/study_data/_save/_npy/keras49_2_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_2_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_2_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_2_test_y.npy')


#2. 모델구성 
                                                   
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
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,verbose=1,
                 validation_split=0.2, callbacks=[earlystopping])


# model.save("./_save/keras23_9_load_diabet.h5")
# model = load_model("./_save/keras23_9_load_diabet.h5")

#4. 평가, 예측\
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

# y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)


# loss :  0.12605245411396027
# acc :  0.9853

# dnn
# loss :  0.3424288332462311
# acc :  0.922

# 함수형
# loss :  0.2445783019065857
# acc :  0.9297

# 증폭
# loss :  0.28579485416412354
# accuracy :  0.9583333134651184