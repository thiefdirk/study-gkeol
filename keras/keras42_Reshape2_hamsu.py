from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv1D, Reshape, LSTM, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
import time
start = time.time()


###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

###################리세이프#######################
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################

#####################XXXXX스케일러XXXXX######################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#################################################

####################겟더미#######################
# y = pd.get_dummies(y)  #겟더미는 y_predict 할때 np아니고 tf.argmax로 바꾸기
# print(y)
################################################

# ####################원핫인코더###################
# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)
# ################################################

###################케라스########################
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))   # y의 라벨값 :  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)
################################################


# 맹그러바바바
# acc 0.98이상
# cifar는 칼라 패션은 흑백


#2. 모델구성
model = Sequential()
model.add(Conv2D(20,3,padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(25, 2))
model.add(MaxPooling2D())
model.add(Conv2D(25, 2,padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Reshape((10,1)))
model.add(Conv1D(10,1, activation='relu'))
model.add(LSTM(16))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()  
# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=30, batch_size=150,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)


acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)
print("time :", time.time() - start)
# loss :  [0.04215491563081741, 0.9890000224113464]
# acc스코어 :  0.989

# loss :  [0.08820876479148865, 0.9760000109672546]
# acc스코어 :  0.976
# time : 354.7991614341736

# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 28, 28, 64)        640
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 32)        8224
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 6, 6, 32)          0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 6, 6, 32)          0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 6, 6, 7)           903
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 3, 3, 7)           0
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 3, 3, 7)           0
# _________________________________________________________________
# flatten (Flatten)            (None, 63)                0
# _________________________________________________________________
# dense (Dense)                (None, 100)               6400
# _________________________________________________________________
# tf.math.multiply (TFOpLambda (None, 100)               0
# _________________________________________________________________
# tf.__operators__.add (TFOpLa (None, 100)               0
# _________________________________________________________________
# activation (Activation)      (None, 100)               0
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                1010
# _________________________________________________________________
# tf.math.multiply_1 (TFOpLamb (None, 10)                0
# _________________________________________________________________
# tf.__operators__.add_1 (TFOp (None, 10)                0
# _________________________________________________________________
# activation_1 (Activation)    (None, 10)                0
# _________________________________________________________________
# reshape (Reshape)            (None, 10, 1)             0
# _________________________________________________________________
# conv1d (Conv1D)              (None, 8, 10)             40
# _________________________________________________________________
# lstm (LSTM)                  (None, 16)                1728
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                170
# _________________________________________________________________
# tf.math.multiply_2 (TFOpLamb (None, 10)                0
# _________________________________________________________________
# tf.__operators__.add_2 (TFOp (None, 10)                0
# _________________________________________________________________
# activation_2 (Activation)    (None, 10)                0
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 10)                0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                110       
# =================================================================
# Total params: 19,225
# Trainable params: 19,225
# Non-trainable params: 0
# _________________________________________________________________