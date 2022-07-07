import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
#1. 데이터

datasets = load_breast_cancer()

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
    )

scaler = RobustScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))


#2. 모델구성
model = Sequential()
model.add(Dense(10, activation= 'linear', input_dim=30))
model.add(Dense(80, activation= 'sigmoid'))
model.add(Dense(90, activation= 'linear'))
model.add(Dense(25, activation= 'relu'))        # relu 강력한놈
model.add(Dense(85, activation= 'linear'))
model.add(Dense(25, activation= 'linear'))      # linear = 기본값 / 생략 가능(회귀모델) 
model.add(Dense(1, activation= 'sigmoid'))      # sigmoid = 0~1 사이로 숫자를 축소해줌. 아웃풋에 sigmoid 입력.
                                                # 회귀모델은 output = linear 자연수치 그데로 나와야 함. 디폴트.
                                                # * 분류모델은 이진 > 마지막 activation = sigmoid 


import time

#3 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam',
              metrics=['accuracy','mse'],)                      # * 이진분류 할 때 binary_crossentropy 반올림.
                                                                # 회귀 - mse,mae ~ / 이진 binary_crossentropy
                                                                # 분류모델 loss에 accuracy(정확도) 같이씀.
                                                                # 2개 이상은 list           
                                                                # 'mse'는 분류모델에서는 잘 맞지 않는다. 
                                                                # 회귀모델 > mitrics=['mae']
                                                                # 분류모델 > metrics=['accuracy','mse']) 
                                                                
from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping =EarlyStopping(monitor='loss', patience=50, mode='min', 
              verbose=1, restore_best_weights = True)          
            

start_time = time.time()

hist = model.fit(x_train, y_train, epochs =500, batch_size = 30, 
                 verbose=1, 
                 validation_split = 0.2,
                 callbacks = [earlystopping])      # callbacks으로 불러온다 erlystopping   

end_time = time.time() - start_time


#4 평가 예측


print('====================')
print(hist)                         #<keras.callbacks.History object at 0x0000013FEE7CFDC0>
print('====================')
print(hist.history)  
print('====================')
print(hist.history['loss'])         # 키 벨류 안에 있는    loss로 양쪽에 '' 을 포함 시킨다. 
print('====================')
print(hist.history['val_loss'])  

print("걸린시간 : ", end_time)
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
# # r2 = r2_score(y_test,y_predict)                         #회귀모델 / 분류모델에서는 r2를 사용하지 않음 
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)
# # print(y_predict)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("걸린시간 : ", end_time)
print('r2 스코어 :', r2)

# import matplotlib.pyplot as plt

# from matplotlib import font_manager, rc
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
# rc('font', family=font_name)                             # << plt 이름오류 해결코드

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker = '.', c ='red', label= 'loss')   # x빼고 y만 넣어주면 됨(순차적).
# plt.plot(hist.history['val_loss'], marker = '.', c ='blue', label= 'val_loss')  
# plt.grid()
# plt.title('제목')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend()
# plt.show()



#1. scaler 하기전 
# accuracy: 0.9035 - mse: 0.0817
# r2 스코어 : 0.6581799808989622
# 걸린시간 :  13.107128381729126

#2. minmaxscaler
# r2 스코어 : 0.9178663511837595
#  accuracy: 0.9708 - mse: 0.0188

#3. standardscaler 
# r2 스코어 : 0.8573367398099374
# accuracy: 0.9649 - mse: 0.0327

#4. MaxAbsScaler
# loss :  [0.06489069759845734, 0.9766082167625427, 
# 0.018419893458485603]
# 걸린시간 :  38.8590350151062
# r2 스코어 : 0.9197293475700395

#5. RobustScaler
# loss :  [0.159059539437294, 0.988304078578949, 0.007930148392915726]
# 걸린시간 :  45.42396950721741
# r2 스코어 : 0.9654418079712949


