# 같으면 0 틀리면 1 (행렬계산)
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#.1 데이터 

x_data =[[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#. 2모델 
model = Sequential()

model.add(Dense(1,input_dim = 2))
model.add(Dense(10))
model.add(Dense(10,activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3.훈련
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['acc'],)
model.fit(x_data,y_data,batch_size=1, epochs=100)

# 4.예측

y_predict = model.predict(x_data)
print(x_data,"결과",y_predict)


results =model.evaluate(x_data,y_data)
print("model.score :",results[1])

# acc = accuracy_score(y_data,y_predict)
# print('acc:',acc)