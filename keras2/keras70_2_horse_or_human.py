from matplotlib.pyplot import hist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.model_selection import train_test_split


x_train = np.load('d:/study_data/_save/_npy/keras47_1_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_1_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_1_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')

print(x_train)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

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

#3. 컴파일

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])
hist = model.fit(x_train, y_train, epochs=200, validation_split=0.3, verbose=2 )
                  

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

# loss :  2.0596190565221918e-14
# val_loss :  24.834657669067383
# accuracy :  1.0
# val_accuracy :  0.5

# VGG16
# loss :  9.190500790956513e-12
# val_loss :  12.051128387451172
# accuracy :  1.0
# val_accuracy :  0.5
# 1/1 [==============================] - 0s 273ms/step - loss: 10.3885 - accuracy: 0.6000
# predict :  [1.]