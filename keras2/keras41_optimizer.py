import numpy as np
import warnings 
warnings.filterwarnings(action='ignore') 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000,input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
# from keras.optimizers import Adam,Adadelta,Adagrad,Adamax,RMSprop,SGD,Nadam
from tensorflow.python.keras.optimizer_v2 import adam, adadelta,adagrad,adamax,rmsprop,nadam

learning_rate = 0.0001

# optimizer1 = adam.Adam(lr=learning_rate )            # loss: 2.5764 lr: 0.001 결과: [[11.669696]]
# optimizer2 = adadelta.Adadelta(lr=learning_rate )    # loss: 2.9061 lr: 0.001 결과: [[10.365168]]
# optimizer3 = adagrad.Adagrad(lr=learning_rate )      # loss: 2.4779 lr: 0.001 결과: [[10.500922]]
# optimizer4 = adamax.Adamax(lr=learning_rate )        # loss: 2.4894 lr: 0.001 결과: [[10.227734]]
# optimizer5 = rmsprop.RMSprop(lr=learning_rate )      # loss: 6.4987 lr: 0.001 결과: [[7.341303]]
# optimizer6 = nadam.Nadam(lr=learning_rate )          # loss: 2.9608 lr: 0.001 결과: [[8.920817]]

optimizers = [adam.Adam(lr=learning_rate) ,adadelta.Adadelta(lr=learning_rate ),adagrad.Adagrad(lr=learning_rate ),
              adamax.Adamax(lr=learning_rate) ,rmsprop.RMSprop(lr=learning_rate ) ,nadam.Nadam(lr=learning_rate ) ]
aa = []
for i in optimizers :
    model.compile(loss='mse',optimizer = i)
    model.fit(x,y,epochs=50,batch_size=1,verbose=0)
    
    loss = model.evaluate(x,y)
    y_predict = model.predict([11])
    
    print('loss:',round(loss,4),i,':',learning_rate ,'결과:',y_predict)
    
    aa.append(y_predict)
    print(aa)
exit()

[array([[10.811256]], dtype=float32),
 array([[10.851852]], dtype=float32), 
 array([[10.940812]], dtype=float32), 
 array([[10.694097]], dtype=float32), 
 array([[13.11612]], dtype=float32), 
 array([[9.92604]], dtype=float32)]

# model.compile(loss='mse',opitimizer ='adam')
model.compile(loss='mse',optimizer = optimizer)

model.fit(x,y,epochs=50,batch_size=1)

#4. 평가,예측
loss = model.evaluate(x,y)
y_predict = model.predict([11])
print('loss:',round(loss,4),'lr:',learning_rate ,'결과:',y_predict)
