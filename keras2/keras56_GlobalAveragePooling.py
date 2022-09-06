from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
print(tf.__version__)

 
#1. 데이터

(x_train,y_train),(x_test,y_test) = mnist.load_data()


x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.


from keras.utils import to_categorical

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델 

drop=0.5
optimizer ='adam'
activation='relu'

inputs = Input(shape=(28,28,1),name='input')
x = Conv2D(64,(2,2), padding='valid',
           activation=activation,name='hidden1')(inputs)    #27,27,128
x = Dropout(drop)(x)
# x = Conv2D(64,(2,2), padding='same',                        #27,27,64
#            activation=activation,name='hidden2')(x)
# x = Dropout(drop)(x)
x = MaxPool2D(2,2)(x)
x = Conv2D(32,(3,3), padding='valid',                       #25,25,32
           activation=activation,name='hidden3')(x)
x = Dropout(drop)(x)

# x = Flatten()(x)                                              # (None,25*25*32) =20000
x = GlobalAveragePooling2D()(x)
# flatten에 연산량이 많아진다는 문제를 해결하는 방법  / 평균으로 뽑아낸다 
x = Dense(100, activation=activation,name='hidden4')(x)
x = Dropout(drop)(x)

outputs = Dense(10, activation='softmax',name ='outputs')(x)

model= Model(inputs=inputs, outputs=outputs)

model.summary()


model.compile(optimizer=optimizer,metrics=['acc'],
                loss='sparse_categorical_crossentropy')
    

import time
start = time.time()
model.fit(x_train,y_train, epochs=5, batch_size=128,validation_split=0.4)
end = time.time()-start


loss,acc = model.evaluate(x_test,y_test)

print('model.score:',model.score)
from sklearn.metrics import accuracy_score

# print("스코어 :", model.score(x_test, y_test))

# x_test = np.argmax(x_test, axis=1)
# y_pred = np.argmax(model.predict(x_test), axis=1)
# y_test = np.argmax(y_test, axis=1)

y_predict = model.predict(x_test)
y_predict = np.argmax(model.predict(x_test),axis=1)
y_test =np.argmax(y_test)
print('acc',accuracy_score(y_test,y_predict))
print('걸린시간',end)

# model.best_params_: {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 1000, 'activation': 'elu'}
# model.best_estimator_: <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001523D7DEBE0>
# model.best_score_: 0.9410666624704996
# model.score: <bound method BaseSearchCV.score of RandomizedSearchCV(cv=3,
#                    estimator=<keras.wrappers.scikit_learn.KerasClassifier object at 0x0000015233DC9CA0>,
#                    n_iter=5,
#                    param_distributions={'activation': ['relu', 'linear',
#                                                        'sigmoid', 'selu',
#                                                        'elu'],
#                                         'batch_size': [1000, 2000, 3000, 4000,
#                                                        5000],
#                                         'drop': [0.3, 0.4, 0.5],
#                                         'optimizer': ['adam', 'rsprop',
#                                                       'adadelta']},
#                    verbose=1)>
# 313/313 [==============================] - 0s 1ms/step
# acc 0.9558
# 걸린시간 16.860023498535156
