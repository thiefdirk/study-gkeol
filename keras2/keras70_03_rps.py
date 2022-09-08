from warnings import filters
# import DenseNet264
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, GlobalAveragePooling2D
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
start = time.time()
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################
#1. 데이터

x_train = np.load('d:/study_data/_save/_npy/keras49_8_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_8_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_8_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_8_test_y.npy')

print(x_train.shape,y_train.shape) #(2064, 100, 100, 3) (2064, 3)
print(x_test.shape,y_test.shape) #(756, 100, 100, 3) (756, 3)

#2. 모델

base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
# model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=80, validation_split=0.2, callbacks=[es, lr], verbose=1)
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

# loss:  [0.013227352872490883, 0.9960317611694336]
# acc스코어 :  1.0
# time : 38.17850875854492

# loss :  6.273979124671314e-06
# acc :  1.0