from email.mime import base
from keras.models import Model
from keras.layers import Dense,Flatten,Input,GlobalAveragePooling2D
from keras.applications import VGG16,InceptionV3
from keras.datasets import cifar100
import numpy as np
(x_train,y_train),(x_test,y_test) = cifar100.load_data()

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

base_model = InceptionV3(include_top=False)
base_model.summary()

print(len(base_model.weights))      # 376

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)

output1 = Dense(100,activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=output1)

# for layer in base_model.layers:    # base_model.layer[3]
    # layer.trainable = False

base_model.tarinable = False    

model.summary()
print(base_model.layers)

# Total params: 24,003,460
# Trainable params: 2,200,676
# Non-trainable params: 21,802,784

model.compile(optimizer='adam',metrics=['acc'],
                loss='categorical_crossentropy')
    
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss',patience=20,mode='min',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

start = time.time()
model.fit(x_train,y_train, epochs=10, batch_size=256,validation_split=0.2,callbacks=[es,reduced_lr])
end = time.time()-start

loss,acc = model.evaluate(x_test,y_test)

# print('model.score:',model.score)
from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
# y_predict = np.argmax(model.predict(x_test),axis=1)
# y_test =np.argmax(y_test)
print('걸린시간',end)
print('loss',loss)
print('acc',acc)