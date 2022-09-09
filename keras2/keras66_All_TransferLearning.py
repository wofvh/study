from keras.applications import VGG16,VGG19
from keras.applications import ResNet50, ResNet152V2
from keras.applications import ResNet101, ResNet101V2, ResNet152
from keras.applications import DenseNet121, DenseNet169
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small,MobileNetV3Large
from keras.applications import NASNetLarge,NASNetMobile
from keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB7
from keras.applications import Xception
# shapea 문제 DenseNet201/ DenseNet169
from gc import callbacks
from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
print(tf.__version__)


#1. 데이터

# (x_train,y_train),(x_test,y_test) = cifar10.load_data()

# x_train = x_train.reshape(50000,32*32*3)
# x_test = x_test.reshape(10000,32*32*3)

# from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(50000,32,32,3)
# x_test = x_test.reshape(10000,32,32,3)

# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

import numpy as np
from sklearn.cluster import k_means
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten
from keras.applications import VGG16

list = [VGG16,VGG19,ResNet50,ResNet152V2,ResNet101, ResNet101V2, ResNet152,  InceptionV3, InceptionResNetV2, DenseNet169,
        MobileNet, MobileNetV2,MobileNetV3Small,MobileNetV3Large,NASNetLarge,NASNetMobile,EfficientNetB0,EfficientNetB1,EfficientNetB7,Xception]
ad = []
ad2 =[]
for i in list : 

    models = i()


    # model = VGG16()
    # model.trainable = False  
    # # vgg16.trainable=False
    # vgg16.summary()

    model =Sequential()
    model.add(models)
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(10,activation='softmax'))

    # model.trainable = False

    model.summary()
                                                # trainable: True / VGG False / model False
    print(len(model.weights))                   # 30                / 30            / 30         
    print(len(model.trainable_weights))         # 30                / 4             /  0

    print(model.layers)

    # import pandas as pd
    # pd.set_option('max_colwidth',-1)
    # layers =[(layer,layer.name,layer.trainable)for layer in model.layers]
    # results = pd.DataFrame(layers,columns=['Layer Type','Layer Name','Layer Trainable'])
    # print(results)

    # model.compile(optimizer='adam',metrics=['acc'],
    #                 loss='categorical_crossentropy')

    

    # import time
    # from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
    # reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

    # start = time.time()
    # model.fit(x_train,y_train, epochs=1, batch_size=256,validation_split=0.2,callbacks=[es,reduced_lr])
    # end = time.time()-start

    # loss,acc = model.evaluate(x_test,y_test)

    # # print('model.score:',model.score)
    # from sklearn.metrics import accuracy_score

    # y_predict = model.predict(x_test)
    # # y_predict = np.argmax(model.predict(x_test),axis=1)
    # # y_test =np.argmax(y_test)
    # print('걸린시간',end)
    # print('loss',loss)
    # print('acc',acc)
    
    print('----------------------------')
    print('모델명:',i.__name__ )
    print('전체 가중치 갯수:',len(model.weights))
    print('훈련가능 가중치 갯수:',len(model.trainable_weights))
    ad.append(i.__name__)
    # ad.append(acc)
    ad.append(len(model.weights))
    ad2.append(i.__name__)
    ad2.append(len(model.trainable_weights))

print('전체 가중치 갯수:',ad)
print('훈련가능 가중치 갯수:',ad2)

# 전체 가중치 갯수: ['VGG16', 36, 'VGG19', 42, 'ResNet50', 324, 'ResNet152V2', 820, 'ResNet101', 630,
#             'ResNet101V2', 548, 'ResNet152', 936, 'InceptionV3', 382, 'InceptionResNetV2', 902,
#             'DenseNet169', 850, 'MobileNet', 141, 'MobileNetV2', 266, 'MobileNetV3Small', 214,
#             'MobileNetV3Large', 270, 'NASNetLarge', 1550, 'NASNetMobile', 1130, 'EfficientNetB0',318,
#             'EfficientNetB1', 446, 'EfficientNetB7', 1044, 'Xception', 240]

# 훈련가능 가중치 갯수: ['VGG16', 36, 'VGG19', 42, 'ResNet50', 218, 'ResNet152V2', 518, 'ResNet101',
#               422, 'ResNet101V2', 348, 'ResNet152', 626, 'InceptionV3', 194, 'InceptionResNetV2', 494,
#               'DenseNet169', 512, 'MobileNet', 87, 'MobileNetV2', 162, 'MobileNetV3Small', 146,
#               'MobileNetV3Large', 178, 'NASNetLarge', 1022, 'NASNetMobile', 746, 'EfficientNetB0', 217,
#               'EfficientNetB1', 305, 'EfficientNetB7', 715, 'Xception', 160]