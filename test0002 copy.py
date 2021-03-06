from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from transformers import AutoTokenizer, pipeline

#1. 데이터
datasets = pd.read_csv("D:\study_data\_temp/marrage.csv", names=["topic", "quote"])

print(datasets)

datasets = pd.DataFrame(datasets)

print(datasets.head(3))

x_data = datasets["topic"]
y_data = datasets["quote"]
print(type(x_data), type(y_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_data = np.array(x_data)
y_data = np.array(y_data)
print(type(x_data), type(y_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>


# 긍정 1, 부정 0
# labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) # (14,)

token = Tokenizer()
token.fit_on_texts(x_data)
token.fit_on_texts(y_data)
# {'너무': 1, '참': 2, '재미없다': 3, '정말': 4, '재밋어요': 5, '최고
# 에요': 6, '잘': 7, '만든': 8, '영화에요': 9, '추천하고': 10, '싶은': 11,
# '영화입니다': 12, '한': 13, '번': 14, '더': 15, '보고': 16, '싶
# 네요': 17, '글세요': 18, '별로에요': 19, '생각보다': 20, '지루해요': 21,
# '연기가': 22, '어색해요': 23, '재미없어요': 24, '재밋네요': 25, '민수가': 26,
# '못': 27, '생기긴': 28, '했어요': 29, '안결': 30, '혼해요': 31,
# '나는': 32, '형권이가': 33}

x = token.texts_to_sequences(x_data)
y = token.texts_to_sequences(y_data)
print(x)
print(y)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17],
#  [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]

print(max(len(i) for i in y)) 
max_len=15
from keras.preprocessing.sequence import pad_sequences # 0채워서 와꾸맞춰줌
# pad_x = pad_sequences(x, padding='pre', maxlen=5) #'post'도있음, 뒤 / truncating= 잘라내기 
pad_y = pad_sequences(y, padding='pre', maxlen=max_len, truncating='pre') #'post'도있음, 뒤 / truncating= 잘라내기 

print(pad_y, pad_y.shape) # (44, 43)
word_size = len(token.word_index)+1 
x = np.array(x)
print(x, x.shape) # (44, 424)
print(pad_y, pad_y.shape) # (44, 15, 424)
# pad_y = pad_y.reshape(44,15,424)
# x = x.reshape(44,1,424)
print(x, x.shape) # (44, 424)
print(pad_y, pad_y.shape) # (44, 15, 424)
# print(pad_test_set, pad_test_set.shape) # (1, 5)

# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  1  5  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25 26 27 28]
#  [ 0  0  0 29 30]] 

# len(x,word_index) x의 인덱스 길이, 수
print("word_size : ", word_size)   # 단어사전의 갯수 : 423

print(np.unique(pad_y, return_counts=True))

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 
# 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), 
#  array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 
#  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],     
#       dtype=int64))

#2. 모델



model = Sequential()
                    #단어사전의 갯수  
# model.add(Embedding(input_dim=31, output_dim=11, input_length=5)) #embedding 에선 아웃풋딤이 뒤로 들어감
# model.add(Embedding(input_dim=31, output_dim=10)) # 인풋렝쓰는 모를 경우 안넣어줘도 자동으로 잡아줌
# model.add(Embedding(31, 10))
# model.add(Embedding(31, 10, 5)) # error
model.add(Embedding(word_size, 20))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(word_size, activation='softmax'))
model.summary()

# Model: "sequential"
# _________________________________________________________________   
# Layer (type)                 Output Shape              Param #      
# =================================================================   
# embedding (Embedding)        (None, 5, 20)             680
# _________________________________________________________________   
# lstm (LSTM)                  (None, 5, 32)             6784
# _________________________________________________________________   
# lstm_1 (LSTM)                (None, 5, 32)             8320
# _________________________________________________________________   
# lstm_2 (LSTM)                (None, 32)                8320
# _________________________________________________________________   
# dense (Dense)                (None, 1)                 33
# =================================================================   
# Total params: 24,137
# Trainable params: 24,137
# Non-trainable params: 0
# _________________________________________________________________   
# Epoch 1/200

#3. 컴파일, 훈련

es = EarlyStopping(monitor='loss', patience=500, mode='auto', verbose=1, 
                              restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, pad_y, epochs=2000, batch_size=64 ,callbacks=[es])

#4. 평가, 예측

def sentence_generator(model, token, current_word, max_len, n) :
    init_word=current_word
    sentence=''
    
    for _ in range(n) : 
        encoded=token.texts_to_sequences([current_word])
        encoded=pad_sequences(encoded, maxlen=max_len, padding='pre')
        
        predicted=model.predict(encoded)
        predicted=np.argmax(predicted, axis=1)
        
        for word, index in token.word_index.items():
            if index==predicted:
                break
            
            
        current_word=current_word+' '+word
        
        sentence=sentence+' '+word
        
    sentence=init_word+sentence
    return sentence

print(sentence_generator(model, token, 'i', max_len, 10))        