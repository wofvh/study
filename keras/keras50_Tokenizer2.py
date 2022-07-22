from keras.preprocessing.text import Tokenizer
import numpy as np

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 이재근이다. 멌있다. 또 또 얘기해봐'


token = Tokenizer()
token.fit_on_texts([text1,text2])

print(token.word_index)             # 인덱스를 부여한다. # 반복횟수가 많을 수록 앞으로 나온다. 


# {'마구': 1, '나는': 2, '매우': 3, '또': 4, '진짜': 5, '맛
# 있는': 6, '밥을': 7, '엄청': 8, '먹었다': 9, '지구용사': 
# 10, '이재근이다': 11, '멌있다': 12, '얘기해봐': 13}

x = token.texts_to_sequences([text1,text2])
print(x)                            # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x_new= x[0] + x[1]
print(x_new)
# [2, 5, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9, 2, 10, 11, 12, 4, 4, 13]
# x_new= to_categorical(x_new)
# print(x_new)
# print(x_new.shape)

x_new = np.array(x_new).reshape(-1,1)
ohe = OneHotEncoder(sparse=True)
x = ohe.fit_transform(x_new).toarray()
print(x)

# ohe = OneHotEncoder(sparse=False)
# x = ohe.fit_transform(x.reshape(-1,1))
# print(x)
