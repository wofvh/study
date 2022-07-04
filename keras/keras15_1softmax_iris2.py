import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


import tensorflow as tf
tf.random.set_seed(66)

########################################
# import numpy as np
# import pandas as pd

# load dataset
# from sklearn.datasets import load_iris
# iris = load_iris()

# target = iris['target']

# num = np.unique(target, axis=0)
# num = num.shape[0]


# encoding = np.eye(num)[target]

# print(encoding)
# y = encoding

# print(y)
############################################

#1. 데이터

datasets = load_iris()
print(datasets.DESCR)                       # (150, 4)
print(datasets.feature_names)

x = datasets ['data']
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)                      # (150, 4)


print('y의 고유값 :', np.unique(y))          # y의 라벨값 : y의 고유값 : [0 1 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)




print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=58525
                                                    )
print(y_train)
print(y_test)



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim= 4))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(15, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# categorical_crossentropy / activation='softmax 다중분류에 사용함.
# softmax는 output에만 적용가능하다. 


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=3, batch_size=100, verbose=1,
                 validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('accuracy : ', results[1])

from sklearn.metrics import accuracy_score


y_predict = np.argmax(y_predict, axix =1)

print(y_predict)

y_test = np.argmax(y_test, axix =1)
print(y_test)


acc = accuracy_score(y_test, y_predict)

print('loss : ' , loss)
print('accuracy : ', acc)


# tensorflow.kercas to preprossing , sklearn에서 제공하는 One-hot encoding 알아보기.



# loss : 0.020010454580187798
# accuracy :  1.0


# print("=================================")

# y_pridict = model.predict(x_test)
# y_pridict = np.argmax(y_pridict, axis=1)

# y_pridict = to_categorical(y_pridict)

# acc = accuracy_score(y_test,y_pridict)
# print('acc :', acc)

# print(y_pridict)
# print(y_test)

# print(y_test[:5])
# print("=================================")
# y_pred = model.predict(x_test[:5])
# print(y_pred)

# y_pridict = 



# print("=================================")