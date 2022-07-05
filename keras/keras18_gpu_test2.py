import numpy as np
import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('돈다') 
    aaa = 'gpu'
else : 
    print('안돈다') 
    bbb = 'cpu'

import numpy as np
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_breast_cancer()
print(datasets)                                   #  행Number of Instances: 569  
                                                  #  열Number of Attributes: 30 
print(datasets.DESCR)

print(datasets.feature_names)

x = datasets.data                       #(569, 30)
y = datasets.target                     #(569,)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation= 'linear', input_dim=30))
model.add(Dense(800, activation= 'relu'))
model.add(Dense(900, activation= 'relu'))
model.add(Dense(205, activation= 'relu'))        # relu 강력한놈
model.add(Dense(805, activation= 'linear'))
model.add(Dense(205, activation= 'linear'))      # linear = 기본값 / 생략 가능(회귀모델) 
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
earlystopping =EarlyStopping(monitor='loss', patience=200, mode='min', 
              verbose=1, restore_best_weights = True)          
            

start_time = time.time()

hist = model.fit(x_train, y_train, epochs =100, batch_size = 5, 
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

print(bbb,"걸린시간 :",end_time)

#y_predict = model.predict(x_test)

# 걸린시간 : 9.569949388504028
# 걸린시간 : 9.552104949951172


#gpu 걸린시간 : 54.20289611816406
'''''


# from sklearn.metrics import r2_score, accuracy_score
# # r2 = r2_score(y_test,y_predict)                         #회귀모델 / 분류모델에서는 r2를 사용하지 않음 
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)
print(y_predict)


import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)                             # << plt 이름오류 해결코드

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', c ='red', label= 'loss')   # x빼고 y만 넣어주면 됨(순차적).
plt.plot(hist.history['val_loss'], marker = '.', c ='blue', label= 'val_loss')  
plt.grid()
plt.title('제목')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()




# loss :  0.09925220161676407
# 걸린시간 :  29.047028303146362
# r2 스코어 : 0.5845791785548424

# loss :  0.07886183261871338
# 걸린시간 :  32.08338141441345
# r2 스코어 : 0.6699231715644725

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlystopping =EarlyStopping(monitor='loss', patience=300, mode='min', 
#               verbose=1, restore_best_weights = True)     
            

# start_time = time.time()

# hist = model.fit(x_train, y_train, epochs =20000, batch_size = 30, 
#                  verbose=1, validation_split = 0.2,
#                  callbacks = [earlystopping])

# activation 0과 1사이에 한정시킨다.0 or 1 로만 정의  1,0 만 가능 
# sigmoid 0과 1사이에 한정시킨다.   0 ~ 1 까지 정의  0.7,0.5,0.3 가능 
# 레어어에 적용시킨다. 


# loss :  0.10784229636192322
# r2 스코어 : 0.8576890179113299



####
# [과제1. accuracy_score 완성 ]
# [과제2. boston, california, diabet,ddareuge, bike, house ]

'''