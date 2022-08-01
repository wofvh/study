import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
season = np.load('d:/study_data/_save/_npy/personalpj_project24.npy')
x_train = np.load('d:/study_data/_save/_npy/project_train11_x.npy')
y_train = np.load('d:/study_data/_save/_npy/project_train11_y.npy')
x_test = np.load('d:/study_data/_save/_npy/project_test11_x.npy')
y_test = np.load('d:/study_data/_save/_npy/project_test11_y.npy')

print(x_train.shape)            # (2000, 150, 150, 3)
print(y_train.shape)            # (2000,)
print(x_test.shape)             # (550, 150, 150, 3)
print(y_test.shape)             # (550,)

# x_train = x_train.reshape(2000,450,150)
# x_test = x_test.reshape(550,450,150)


from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM,MaxPool2D


#2. 모델 
# model = Sequential()
# model.add(Conv2D(64,(3,3), input_shape = (150,150,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(7,activation='softmax'))

model = Sequential()
model.add(Conv2D(128,(2,2),input_shape=(150,150,3),padding='same',activation='relu'))
# model.add(conv_base)
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
# model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.6))                 #과적합방지
model.add(Dense(7,activation='softmax'))
model.summary()
    
# model.save("./_save/project1_save_model.h2")


#3. 컴파일.훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True)     

hist = model.fit(x_train,y_train, epochs=200,validation_split=0.15,verbose=2,batch_size=50,
                 callbacks=[earlystopping]) 

model.save('C:\study\_save/project4.h5')
# model = load_model('C:\study\_save/project.h5')


#4. 예측
# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ',loss[-1])
# print('accuracy : ', accuracy[-1])

############################################
loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(season)

# y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict,axis=1) 

# y_test = tf.argmax(y_test,axis=1) 

y_test2 = [0,1,2,3,4,5,6]
y_predict = model.predict(season)
y_test = np.argmax(y_test, axis= 1)
y_predict = np.argmax(y_predict, axis=1)
print('predict : ',y_predict)
acc = accuracy_score(y_test2,y_predict)
print('acc : ',acc)

if y_predict[0] == 0:
    print('hail ')
elif  y_predict[0] ==1 :
    print('lighting')
elif  y_predict[0] ==2 :
    print('rain')
elif  y_predict[0] ==3 :
    print('rainbow')
elif  y_predict[0] ==4 :
    print('sunshine')        
elif  y_predict[0] ==5 :
    print('smog')        
else :
    print('snow')   
############################################
# y_predict = model.predict(x_test)
# # y_predict = tf.argmax(y_predict,axis=1) 
# # y_test = tf.argmax(y_test,axis=1) 
# acc = accuracy_score(y_test,y_predict)
# print('acc : ',acc)
############################################

# 0.hail   1.lighting   2.rain   3.rime   4.shine   5.smog   6.snow 

# 0.hail :       70%  [0 0 0 0 3 6 6 0 0 0]

# 1.lighting :   90%  [1 1 1 1 4 1 1 1 1 1]

# 2.rain :       40%  [0 0 2 0 0 2 6 6 2 2]

# 3.rainbow :    60%  [6 3 3 4 5 3 4 3 3 3]

# 4.sunshine :   90%  [4 4 4 4 4 3 4 4 4 4]

# 5.smog :       50%  [5 6 3 3 5 4 5 3 5 5]

# 6.snow :       70%  [6 6 0 5 6 6 6 6 0 6]


