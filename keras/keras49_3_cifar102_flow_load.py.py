# 넘파이에서 불러와서 모델구성
# 성능비교



from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D #Flatten평평하게해라.  # 이미지 작업 conv2D 
from keras.datasets import mnist, cifar10 , fashion_mnist
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 


#1. 데이터
x_train = np.load('d:/study_data/_save/_npy/keras49_3_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_3_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_3_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_3_test_y.npy')


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# x_train = x_train.reshape(50000, 32,96)
# x_test = x_test.reshape(10000, 32,96)



x_train = x_train.reshape(40000, 32,96)
x_test = x_test.reshape(10000, 32,96)
print(x_train)
print(x_test)
# 원핫인코딩 

#2. 모델구성 

                                    
model = Sequential()    
model.add(Conv1D(1280, 2, input_shape=(32,96)))
model.add(Flatten())
model.add(Dense(640, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax')) 


#3. 컴파일 구성 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=70, mode='auto', 
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=200, batch_size=1000,verbose=2,
                 validation_split=0.3, callbacks=[earlystopping])


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
print('conv1D cifar10')

# loss :  4.3395161628723145
# acc :  0.519

# loss :  3.9987804889678955
# acc :  0.600
# 함수형
# loss :  1.5904154777526855
# acc :  0.4591


# LSTM
#  accuracy: 0.0749
# loss :  5.423413276672363
# acc :  0.0749

# loss :  1821.7581787109375
# acc :  0.4357

