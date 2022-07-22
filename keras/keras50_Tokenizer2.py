from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text1 = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'
text2 = '나는 지구용사 이재근이다. 멋있다. 또 얘기해봐.'

token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'마구': 1, '나는': 2, '매우': 3, '진짜': 4, '맛있는': 5, 
# '밥을': 6, '엄청': 7, '먹었다': 8, '지구용사': 9, '이재근이다': 10, '멋있다': 
# 11, '또': 12, '얘기해봐': 13}

x = token.texts_to_sequences([text1, text2])
print(x)
# print(x.shape)
# [[2, 4, 3, 3, 5, 6, 7, 1, 1, 1, 8], [2, 9, 10, 11, 12, 13]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩

x_new = x[0] + x[1]
print(x_new)
# print(x_new.shape)
x_new = np.array(x_new)
x_new = x_new.reshape(17,1)
# x_new = to_categorical(x_new)
ohe = OneHotEncoder(sparse=False)
x_new = ohe.fit_transform(x_new)
print(x_new)
print(x_new.shape)

# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# (17, 13)

