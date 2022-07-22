from keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밋어요','참 최고예요','참 잘 만든 영화예요',
       '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요', '글세요'
       '별로예요','생각보다 지루해요','연기가 어색해요',
       '재미없어요','너무 재미없다','참 재밋네요','민수가 못 생기긴 했어요',
       '안결 혼해요'
       ]

docs2 = ['나는 형권이가 정말 재미없다 너무 정말']

# 긍정1, 부정 0
labels =np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])  # 14
# x 14,5
# y 14, 
token = Tokenizer()
token.fit_on_texts(docs)
token.fit_on_texts(docs2)

print(token.word_index)

# {'참': 1, '너무': 2, '재밌어요': 3, '최고예요': 4, '잘': 
# 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, ' 
# 영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, 
# '싶네요': 15, '글세요별로예요': 16, '생각보다': 17, '지
# 루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 
# 21, '재미없다': 22, '재밋네요': 23, '민수가': 24, '못': 25, 
# '생기긴': 26, '했어요': 27, '안결': 28, '혼해요': 29}

x = token.texts_to_sequences(docs)
print(x)

# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15],
#  [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25, 26, 27], [28, 29]]

from keras.preprocessing.sequence import pad_sequences   # 0을 채운다.
pad_x = pad_sequences(x,padding="pre",maxlen=6,truncating='pre')
# 가장큰 5개 짜리를 기준으로 0을 앞에서 부터 채운다. 
# # padding="pre" 앞에0을 채우겠다. # maxlen=5 최대글자를 다섯글자만.

print(pad_x)
print(pad_x.shape)                        # (13, 6)

word_size= len(token.word_index)
print("word_size:",word_size)             # word_size: 32

# print(np.unique(pad_x,return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]), 
#  array([33,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
#         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=int64))

#2. 모델 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Embedding ,Input # input layer에서 Embedding 사용. 

# model = Sequential()                        # input(13,5)
# # model.add(LSTM(32,))
# # model.add(Embedding(input_dim=33, output_dim=10,input_length=5))   # input_dim = 단어사전개수
# # model.add(Embedding(input_dim=31, output_dim=10))   # input_dim = 단어사전개수  / output_dim = output
# # model.add(Embedding(31,10))
# # model.add(Embedding(31,10,5))                    # input_length =5는  input_length 포함해야한다. 아니면 생략 (자동지정.)
# model.add(Embedding(20,11,input_length=6))      # input_length =5는  input_length 포함해야한다. 아니면 생략 (자동지정.)
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))
input1 = Input(shape=(6,))       # 컬럼3개를 받아드린다.
dense1 = Embedding(input_dim=33, output_dim=10,input_length=5)(input1)    # 원핫 필요없이 사용할수 있는 모델
dense2 = Dense(10)(dense1)         # Dense 뒤에 input 부분을 붙여넣는다.
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(30, activation='sigmoid')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)

model = Model(inputs = input1, outputs = output1)

model.summary()

#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(pad_x,labels,epochs=100,batch_size=16)



#4. 평가 
acc = model.evaluate(pad_x,labels)[1]
print('aac:',acc)

# 결과 ? 긍정 ? 부정? 
# 개수 
y_predict = ['나는 형권이가 정말 재미없다 너무 정말']
# token.fit_on_texts(y_predict)
print(token.word_index)
# y = token.texts_to_sequences(y_predict)
# print(y)

y1_predict = model.predict(y_predict)
print('predict : ',np.around(y1_predict[-1]))
