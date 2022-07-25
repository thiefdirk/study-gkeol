# -*- coding: utf-8 -*-
# Char-RNN 예제
# Author : solaris33
# Project URL : http://solarisailab.com/archives/2487
# GitHub Repository : https://github.com/solaris33/char-rnn-tensorflow/
# Reference : https://github.com/sherjilozair/char-rnn-tensorflow



from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

#1. 데이터
docs = pd.read_csv('D:\study_data\_temp/marrage.csv')


token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4, '재밋어요': 5, '최고
# 에요': 6, '잘': 7, '만든': 8, '영화에요': 9, '추천하고': 10, '싶은': 11,
# '영화입니다': 12, '한': 13, '번': 14, '더': 15, '보고': 16, '싶
# 네요': 17, '글세요': 18, '별로에요': 19, '생각보다': 20, '지루해요': 21,
# '연기가': 22, '어색해요': 23, '재미없어요': 24, '재밋네요': 25, '민수가': 26,
# '못': 27, '생기긴': 28, '했어요': 29, '안결': 30, '혼해요': 31,
# '나는': 32, '형권이가': 33}

y = token.texts_to_sequences(docs)
print(y)
# print(test_set)