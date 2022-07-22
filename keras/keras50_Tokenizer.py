from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}

x = token.texts_to_sequences([text])
x = np.array(x)
x = x.reshape(11,1)

# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩


ohe = OneHotEncoder(sparse=False)
x = ohe.fit_transform(x)
print(x)
print(x.shape)

# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
# (11, 8)

