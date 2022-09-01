import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
#.1 데이터 

x_data =[[0,0],[0,1],[1,0],[1,1]]
y_data = [0,0,0,1]

#. 2모델 
model = Perceptron()

# 3.훈련

model.fit(x_data,y_data)

# 4.예측

y_predict = model.predict(x_data)
print(x_data,"결과",y_predict)


results =model.score(x_data,y_data)
print("model.score :",results)

acc = accuracy_score(y_data,y_predict)
print('acc:',acc)


