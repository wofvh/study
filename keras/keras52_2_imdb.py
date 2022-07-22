from keras.datasets import imdb
import numpy as np

(x_train,y_train),(x_test,y_test) = imdb.load_data(
    num_words=10000
)

from itertools import count
from tkinter import Y
from keras.datasets import reuters
import numpy as np
import pandas as pd


print(x_train)
print(x_train.shape,x_test.shape)        # (25000,) (25000,)
print(y_train)
print(np.unique(y_train,return_counts = True))      # 46 뉴스카테고리
print(len(np.unique(y_train)))                      # 46

print(type(x_train),type(y_train))      # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]),type(y_train[0]))      # <class 'list'> <class 'numpy.int64'>
# pad_x = pad_sequences(x,padding="pre",maxlen=6,truncating='pre')
print(len(x_train[0]))                  # 218
print(len(x_train[1]))                  # 189
print(len(x_train))                     # 25000

            
print('뉴스기사의 최대길이:',max(len(i) for i in x_train))
# 뉴스기사의 최대길이: 2494
print('뉴스기사의 평균길이:',sum(map(len,x_train))/len(x_train))
# 뉴스기사의 평균길이: 238.71364

#전치리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
# pad_x = pad_sequences(x_trian,padding="pre",maxlen=6,truncating='pre')
x_train = pad_sequences(x_train, padding='pre',maxlen=100,truncating='pre')
print(x_train.shape)            # (25000, 100)
x_test = pad_sequences(x_test, padding='pre',maxlen=100,truncating='pre')
print(x_test.shape)            # (25000, 100)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,y_train.shape)      # (25000, 100) (25000, 2)
print(x_test.shape,y_test.shape)        # (25000, 100) (25000, 2)

#2. 모델 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding  # input layer에서 Embedding 사용. 

model = Sequential()                        # input(13,5)
# model.add(LSTM(32,))
# model.add(Embedding(input_dim=33, output_dim=10,input_length=5))   # input_dim = 단어사전개수
# model.add(Embedding(input_dim=31, output_dim=10))   # input_dim = 단어사전개수  / output_dim = output
# model.add(Embedding(31,10))
# model.add(Embedding(31,10,5))                    # input_length =5는  input_length 포함해야한다. 아니면 생략 (자동지정.)
model.add(Embedding(31,11,input_length=6))      # input_length =5는  input_length 포함해야한다. 아니면 생략 (자동지정.)
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=100,batch_size=10000)



#4. 평가 
acc = model.evaluate(x_test,y_test)
print('aac:',acc)

# 결과 ? 긍정 ? 부정? 
# 개수 
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('predict : ',y_predict[-1])

# loss : 0.31505573
# acc : 0.68667716


