from keras.preprocessing.text import Tokenizer
import numpy as np
text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)             # 인덱스를 부여한다. # 반복횟수가 많을 수록 앞으로 나온다. 

x = token.texts_to_sequences([text])
print(x)                            # [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

# x= to_categorical(x)
# print(x)
# print(x.shape)

x = np.array(x).reshape(-1,1)
ohe = OneHotEncoder(sparse=True)
x = ohe.fit_transform(x).toarray()
print(x)


# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 12, 9)