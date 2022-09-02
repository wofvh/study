import autokeras as ak
print(ak.__version__)
import tensorflow as tf
import keras
import time
from keras.datasets import mnist, cifar10 , fashion_mnist

#1. 데이터
(x_train,y_train),(x_test,y_test) =\
    keras.datasets.cifar10.load_data()
    
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#2. 모델 
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2            #시도횟수
)

#3 컴파일 훈련
start =time.time()
model.fit(x_train,y_train,epochs=5)
end = time.time()
#4. 예측결과
y_predict = model.predict(x_test)
results =model.evaluate(x_test,y_test)
print('걸린시간:',round(end-start,4))

print('결과:',results)
