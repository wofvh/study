import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
season = np.load('d:/study_data/_save/_npy/personalpj_project31.npy')
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
from keras.applications.vgg16 import VGG16
from keras import models, layers
#2. 모델 


pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
pre_trained_vgg.trainable = False
pre_trained_vgg.summary()
additional_model = models.Sequential()
additional_model.add(pre_trained_vgg)
additional_model.add(layers.Flatten())
additional_model.add(layers.Dense(50, activation='relu'))
additional_model.add(layers.Dense(50, activation='relu'))
additional_model.add(layers.Dense(50, activation='relu'))
additional_model.add(layers.Dense(7, activation='softmax'))
additional_model.summary()


# model = Sequential()
# model.add(Conv2D(128,(2,2),input_shape=(150,150,3),padding='same',activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Conv2D(128,(2,2),padding='same',activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Flatten())
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.6))                
# model.add(Dense(7,activation='softmax'))
# model.summary()
    
# model.save("./_save/project1_save_model.h2")


#3. 컴파일.훈련

additional_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=50, mode='auto', 
              verbose=2, restore_best_weights = True)     

hist = additional_model.fit(x_train,y_train, epochs=20  ,validation_split=0.5,verbose=2,batch_size=32,
                 callbacks=[earlystopping]) 

# model.save('C:\study\_save/project4.h5')  # 웹에서 사용하기위해 
# model = load_model('C:\study\_save/project.h5')


#4. 예측
# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ',loss[-1])
# print('accuracy : ', accuracy[-1])

############################################

# y_predict = model.predict(season)

# y_predict = model.predict(x_test)
# y_predict = tf.argmax(y_predict,axis=1) 

# y_test = tf.argmax(y_test,axis=1) 

y_test2 = [0,1,2,3,4,5,6]
y_predict = additional_model.predict(season)
y_test = np.argmax(y_test, axis= 1)
y_predict = np.argmax(y_predict, axis=1)
print('predict : ',y_predict)
acc = accuracy_score(y_test2,y_predict)
print('acc : ',acc)


if y_predict[0] == 0:
    print('<우박>  내륙에는 우박이 떨어지는 곳이 있겠습니다. 각별히 유의하기 바랍니다. ')
elif  y_predict[0] ==1 :
    print('<번개>  풍과 천둥번개가 동반될 수 있습니다. 틈나는 대로 날씨 변화를 점검해주시기 바랍니다.')
elif  y_predict[0] ==2 :
    print('<비> 비구름대가 발달하면서 내륙에는 비가 오는 곳이 있겠습니다.반드시 우산을 챙기시기 바랍니다.')
elif  y_predict[0] ==3 :
    print('<무지개>  소나기가 지나간 하늘에 무지개가 떴습니다.')
elif  y_predict[0] ==4 :
    print('<맑은날> 고기압의 영향으로 대체로 날은 맑겠습니다.미세먼지 농도는 좋음 단계로 야외 활동하기 좋습니다.')        
elif  y_predict[0] ==5 :
    print('<황사> 이번 베이징의 황사는 중국의 황사경보 4단계 중 낮은 청색경보 수준이라, 한국에는 약한 수준의 황사정도가 예상됩니다.') 
elif  y_predict[0] ==6 :
    print('<눈>  찬 대륙고기압이 우리나라에 확장되면서 기온이 급격히 낮아지고 대설이 예상됩니다.')      
    

###########################################
# y_predict = model.predict(x_test)
# # y_predict = tf.argmax(y_predict,axis=1) S
# # y_test = tf.argmax(y_test,axis=1) 
# acc = accuracy_score(y_test,y_predict)
# print('acc : ',acc)
############################################

# 0.hail   1.lighting   2.rain   3.rainbow   4.shine   5.smog   6.snow 

# 0.hail :       70%  [0 0 0 0 3 6 6 0 0 0]

# 1.lighting :   90%  [1 1 1 1 4 1 1 1 1 1]

# 2.rain :       40%  [0 0 2 0 0 2 6 6 2 2]

# 3.rainbow :    60%  [6 3 3 4 5 3 4 3 3 3]

# 4.sunshine :   90%  [4 4 4 4 4 3 4 4 4 4]

# 5.smog :       50%  [5 6 3 3 5 4 5 3 5 5]

# 6.snow :       70%  [6 6 0 5 6 6 6 6 0 6]


# predict :  [5 0 6 3 2 0 1]