from itertools import count
from tkinter import Y
from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train,y_train),(x_test,y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train)
print(x_train.shape,x_test.shape)        # (8982,)  (2246,)
print(y_train)
print(np.unique(y_train,return_counts = True))      # 46 뉴스카테고리
print(len(np.unique(y_train)))                      # 46

print(type(x_train),type(y_train))      # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]),type(y_train[0]))      # <class 'list'> <class 'numpy.int64'>
# pad_x = pad_sequences(x,padding="pre",maxlen=6,truncating='pre')
print(len(x_train[0]))                  # 87
print(len(x_train[1]))                  # 56
print(len(x_train))                     # 8982

            
print('뉴스기사의 최대길이:',max(len(i) for i in x_train))
# 뉴스기사의 최대길이: 2376
print('뉴스기사의 평균길이:',sum(map(len,x_train))/len(x_train))
# 뉴스기사의 평균길이: 145.5398574927633

#전치리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
# pad_x = pad_sequences(x_trian,padding="pre",maxlen=6,truncating='pre')
x_train = pad_sequences(x_train, padding='pre',maxlen=100,truncating='pre')
print(x_train.shape)            # (8982, 100)
x_test = pad_sequences(x_test, padding='pre',maxlen=100,truncating='pre')
print(x_test.shape)            # (2246, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,y_train.shape)      # (8982, 100) (8982, 46)
print(x_test.shape,y_test.shape)        # (2246, 100) (2246, 46)



#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding  # input layer에서 Embedding 사용. 

model = Sequential()                        # input(13,5)
# model.add(LSTM(32,))
# model.add(Embedding(input_dim=33, output_dim=10,input_length=5))   # input_dim = 단어사전개수
# model.add(Embedding(input_dim=31, output_dim=10))   # input_dim = 단어사전개수  / output_dim = output
# model.add(Embedding(31,10))
# model.add(Embedding(31,10,5))                    # input_length =5는  input_length 포함해야한다. 아니면 생략 (자동지정.)
model.add(Embedding(10000,2,input_length=100))      # input_length =5는  input_length 포함해야한다. 아니면 생략 (자동지정.)
model.add(LSTM(32))
model.add(Dense(46, activation='relu'))
model.add(Dense(46, activation='softmax'))
model.summary()

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=20,batch_size=5000)



#4. 평가 
acc = model.evaluate(x_test,y_test)
print('aac:',acc[0])

# 결과 ? 긍정 ? 부정? 
# 개수 
# loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)
# print('predict : ',y_predict[0])