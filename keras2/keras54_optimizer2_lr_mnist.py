# 넘파이에서 불러와서 모델구성
# 성능비교

# 넘파이에서 불러와서 모델구성
# 성능비교
from warnings import filters
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
import datetime
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################

x_train = np.load('d:/study_data/_save/_npy/keras49_2_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_2_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_2_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_2_test_y.npy')



#2. 모델구성
model = Sequential()
input1 = Input(shape=(28,28,1))
conv2D_1 = Conv2D(100,3, padding='same')(input1)
MaxP1 = MaxPooling2D()(conv2D_1)
drp1 = Dropout(0.2)(MaxP1)
conv2D_2 = Conv2D(200,2,
                  activation='relu')(drp1)
MaxP2 = MaxPooling2D()(conv2D_2)
drp2 = Dropout(0.2)(MaxP2)
conv2D_3 = Conv2D(200,2, padding='same',
                  activation='relu')(drp2)
MaxP3 = MaxPooling2D()(conv2D_3)
drp3 = Dropout(0.2)(MaxP3)
flatten = Flatten()(drp3)
dense1 = Dense(200)(flatten)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('relu')(batchnorm1)
drp4 = Dropout(0.2)(activ1)
dense2 = Dense(100)(drp4)
batchnorm2 = BatchNormalization()(dense2)
activ2 = Activation('relu')(batchnorm2)
drp5 = Dropout(0.2)(activ2)
dense3 = Dense(100)(drp5)
batchnorm3 = BatchNormalization()(dense3)
activ3 = Activation('relu')(batchnorm3)
drp6 = Dropout(0.2)(activ3)
output1 = Dense(10, activation='softmax')(drp6)
model = Model(inputs=input1, outputs=output1)   

# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련

from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.001

# optimizer = adam.Adam(lr=learning_rate) # learning_rate default = 0.001
# optimizer = adadelta.Adadelta(lr=learning_rate)
# optimizer = adagrad.Adagrad(lr=learning_rate)
# optimizer = adamax.Adamax(lr=learning_rate)
# optimizer = rmsprop.RMSprop(lr=learning_rate)
optimizer = nadam.Nadam(lr=learning_rate)


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([save_filepath, date, '_', filename])
#                       )
start = datetime.datetime.now()

hist = model.fit(x_train, y_train, epochs=40, batch_size=2000,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict.argmax(axis=1))

# acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

end = datetime.datetime.now()
time = end - start

print('소요 시간 : ', time)
print('loss : ', loss[-1])
print('accuracy : ', acc)


#adam 디폴트
# loss :  0.020334603264927864
# accuracy :  0.9938125014305115

# adam, lr=0.001
# loss :  0.018259568139910698
# accuracy :  0.994362473487854

# Adadelta, lr=0.01
# loss :  0.7895901203155518
# accuracy :  0.7440875172615051

# Adagrad, lr=0.01
# loss :  0.22388452291488647
# accuracy :  0.9306750297546387

# Adamax, lr=0.01
# loss :  0.03864812105894089
# accuracy :  0.9880250096321106

# Adamax, lr=0.001
# loss :  0.03226569667458534
# accuracy :  0.9899125099182129

# RMSprop, lr=0.001
# loss :  0.01462838426232338
# accuracy :  0.9954125285148621

# Nadam, lr=0.001
# loss :  0.01715708337724209
# accuracy :  0.9944999814033508