import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
season = np.load('d:/study_data/_save/_npy/personaltest33.npy')
x_train = np.load('d:/study_data/_save/_npy/project_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/project_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/project_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/project_test_y.npy')

print(x_train.shape)            # (2000, 150, 150, 3)
print(y_train.shape)            # (2000,)
print(x_test.shape)             # (550, 150, 150, 3)
print(y_test.shape)             # (550,)

# x_train = x_train.reshape(2000,450,150)
# x_test = x_test.reshape(550,450,150)


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM


#2. 모델 
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape = (150,150,3), padding='same', activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))
    

#3. 컴파일.훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=50, mode='auto', 
              verbose=1, restore_best_weights = True)     

hist = model.fit(x_train,y_train, epochs=50,validation_split=0.3,verbose=2,batch_size=16,
                 callbacks=[earlystopping]) 


#4. 예측
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-1])
print('accuracy : ', accuracy[-1])
# loss = model.evaluate(x_test, y_test)
# x_predict = model.predict(x_test)

# y_predict = model.predict(x_test)
# # y_predict = np.argmax(y_predict, axis= 1)
# # y_test = np.argmax(y_test, axis= 1)



# season_predict = model.predict(season)
# y_test = np.argmax(y_test, axis= 1)
# y_predict = np.argmax(season_predict, axis=1) 
# print('predict : ',season_predict)
############################################
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(season)

y_test = np.argmax(y_test, axis= 1)
y_predict = np.argmax(y_predict, axis=1) 
print('predict : ',y_predict)

# acc = accuracy_score(y_test,y_predict) 
# print('acc : ', acc) 

# 0.hail   1.lighting   2.rain   3.rime   4.shine   5.smog   6.snow 

# 0.hail :       5/7   [0 3 0 0 0 0 0 2]

# 1.lighting :   6/7  [1 1 1 0 1 1 1]

# 2.rain :       4/7   [2 2 0 2 6 2 6]

# 3.rime :       2/7
# [[0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]]

# 4.sunshine : 6/7
#  [[0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]]

#  5.smog : 3/7
# [[1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0.]]

# 6.snow : 7/7
# [[0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 1.]]
