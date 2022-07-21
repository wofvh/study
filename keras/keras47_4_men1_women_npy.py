# 본인 사진으로 predict하시오 

from matplotlib.pyplot import hist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.model_selection import train_test_split

men = ImageDataGenerator(
    rescale=1./255)

men1 = men.flow_from_directory(
    'D:\study_data\_data\image/bbbb',
    target_size=(100,100),# 크기들을 일정하게 맞춰준다.
    batch_size=10000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )
print(men1[0][0])

np.save('d:/study_data/_save/_npy/keras51_6_men1.npy', arr=men1[0][0])

# #1. 데이터
# x_train = np.load('d:/study_data/_save/_npy/keras47_4_train_x.npy')
# y_train = np.load('d:/study_data/_save/_npy/keras47_4_train_y.npy')
# x_test = np.load('d:/study_data/_save/_npy/keras47_4_test_x.npy')
# y_test = np.load('d:/study_data/_save/_npy/keras47_4_test_y.npy')

# #2 모델구성 
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D

# model = Sequential()
# model.add(Conv2D(64,(2,2), input_shape = (100,100,3),padding='same',activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(48,(3,3),activation='relu'))
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# # model.add(Dropout(0.3))
# model.add(Dense(100,activation='relu'))
# # model.add(Dropout(0.3))
# model.add(Dense(1,activation='sigmoid'))
    

# #3. 컴파일.

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])
# # model.fit(xy_train[0][0],xy_train[0][1])          # 배치를 최대로 잡으면 가능
# hist = model.fit(x_train,y_train, epochs=100,validation_split=0.3,verbose=2) 
#                     # steps_per_epoch=32,  # steps_per_epoch=32 데이터를 batch size로 나눈것. 160/5 =32 
#                     # validation_data=xy_test,
#                     # validation_steps=4)


# #4. 예측 

# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ',loss[-1])
# print('val_loss : ',val_loss[-1])
# print('accuracy : ', accuracy[-1])
# print('val_accuracy : ', val_accuracy[-1])
# print("=========================1.기본출력========================")
# loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)
# print('predict : ',y_predict[-1])

# # loss :  1.1446439884821302e-07
# # val_loss :  4.418682098388672
# # accuracy :  1.0
# # val_accuracy :  0.6389937400817871
# # predict :  [1.]

