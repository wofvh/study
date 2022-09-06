from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.python.keras.optimizer_v2 import adam, adadelta,adagrad,adamax,rmsprop,nadam

import tensorflow as tf
print(tf.__version__)
# 하이퍼 파라미터에 노드 추가, learning_rate 추가
 
#1. 데이터

(x_train,y_train),(x_test,y_test) = mnist.load_data()


x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.


from keras.utils import to_categorical

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. 모델 

def build_model(drop=0.5,optimizer =adam,activation='relu',node=128,lr='lr'):
    
    inputs = Input(shape=(28*28),name='input')
    x = Dense(node,activation=activation,name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation,name='hidden3')(x)
    outputs = Dense(10, activation='softmax',name ='outputs')(x)

    model= Model(inputs=inputs, outputs=outputs)
    optimizer = optimizer(lr)
    model.compile(optimizer=optimizer,metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [1000,2000,3000,4000,5000]
    optimizer = ['adam','rsprop','adadelta']
    dropout = [0.3,0.4,0.5]
    activation = ['relu','linear','sigmoid','selu','elu']
    node = [512,256,128,64]
    lr = [0.1,0.001,0.0001]
    return{'batch_size': batchs, 'optimizer':optimizer,
           'drop':dropout,'activation':activation,'node':node,
           'lr':lr}
    
    
hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [10, 200, 300, 400, 500], 'optimizer': ['adam', 'rsprop', 'adadelta'],
#  'drop': [0.3, 0.4, 0.5], 'activation': ['relu', 'linear', 'sigmoid', 'selu', 'elu']}


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor 
# 사이킷런으로 케라스(텐서플로우)모델 감쌈


keras_model = KerasClassifier(build_fn=build_model, verbose=1)
# GridSearchCV는 케라스 안에있는 모델만 인정하기 때문에 KerasClassifier 로 내가 만든
# 함수를 정의해줘야한다. 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=5,verbose=1)

import time
start = time.time()
model.fit(x_train,y_train, epochs=5, validation_split=0.4)
end = time.time()-start



print('model.best_params_:',model.best_params_)
print('model.best_estimator_:',model.best_estimator_)
print('model.best_score_:',model.best_score_)
print('model.score:',model.score)
from sklearn.metrics import accuracy_score

# print("스코어 :", model.score(x_test, y_test))

# x_test = np.argmax(x_test, axis=1)
# y_pred = np.argmax(model.predict(x_test), axis=1)
# y_test = np.argmax(y_test, axis=1)



y_predict = model.predict(x_test)
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
