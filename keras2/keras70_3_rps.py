from matplotlib.pyplot import hist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.model_selection import train_test_split
x_train = np.load('d:/study_data/_save/_npy/keras47_3_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_3_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_3_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_3_test_y.npy')
# np.save('d:/study_data/_save/_npy/keras47_3_train_x.npy', arr=x_train)
# np.save('d:/study_data/_save/_npy/keras47_3_train_y.npy', arr=y_train)
# np.save('d:/study_data/_save/_npy/keras47_3_test_x.npy', arr=x_test)
# np.save('d:/study_data/_save/_npy/keras47_3_test_y.npy', arr=y_test)

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
model.add(Dense(3,activation='softmax'))
    

#3. 컴파일.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
# model.fit(xy_train[0][0],xy_train[0][1])          # 배치를 최대로 잡으면 가능
hist = model.fit(x_train,y_train, epochs=100,validation_split=0.3,verbose=2) 
                    # steps_per_epoch=32,  # steps_per_epoch=32 데이터를 batch size로 나눈것. 160/5 =32 
                    # validation_data=xy_test,
                    # validation_steps=4)

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

# model.evaluate(x_test,y_test)
# y_predict = model.predict(x_test)
# from sklearn.metrics import r2_score, accuracy_score
# y_predict= np.argmax(y_predict)

# y_test = np.argmax(y_test)

# acc = accuracy_score(y_test,y_predict)
# print('acc:',acc)

# loss :  5.7450240120715534e-09
# val_loss :  5.5170868762388636e-08
# accuracy :  1.0
# val_accuracy :  1.0

# VGG16
# loss :  7.519222577911933e-09
# val_loss :  7.132754831218335e-08
# accuracy :  1.0
# val_accuracy :  1.0
# 16/16 [==============================] - 1s 47ms/step - loss: 3.1458e-08 - accuracy: 1.0000
# predict :  [3.5903454e-29 1.0000000e+00 4.7709406e-27]