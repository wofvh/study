import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
season = np.load('d:/study_data/_save/_npy/personaltest21.npy')
x_train = np.load('d:/study_data/_save/_npy/project_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/project_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/project_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/project_test_y.npy')

print(x_train.shape)            # (2000, 150, 150, 3)
print(y_train.shape)            # (2000,)
print(x_test.shape)             # (550, 150, 150, 3)
print(y_test.shape)             # (550,)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D

#2. 모델 
model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (150,150,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))
    

#3. 컴파일.훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

# earlystopping =EarlyStopping(monitor='loss', patience=50, mode='auto', 
#               verbose=1, restore_best_weights = True)     

hist = model.fit(x_train,y_train, epochs=100,validation_split=0.3,verbose=2,batch_size=16)
                #  callbacks=[earlystopping]) 


#4. 예측
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss)
print('accuracy : ', accuracy)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(season)
print('predict : ',y_predict.round())


# loss :  1.0985032320022583
# val_loss :  1.0987937450408936
# accuracy :  0.3424285650253296
# val_accuracy :  0.3333333432674408
# predict :  [0.24719454 0.05714595 0.6956595 ]


# predict : 
#  [[0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]]