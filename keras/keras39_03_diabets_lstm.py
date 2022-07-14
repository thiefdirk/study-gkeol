from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes
import numpy as np
import time
from pathlib import Path

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################

#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (353, 10) (89, 10)


###################리세이프#######################
x_train = x_train.reshape(353, 5, 2)
x_test = x_test.reshape(89, 5, 2) 
print(x_train.shape)
print(np.unique(y_train, return_counts=True))
#################################################

print(x_train.shape, x_test.shape)  # (353, 5, 2, 1) (89, 5, 2, 1)  

start_time = time.time()

#2. 모델구성

model = Sequential()
model.add(LSTM(200, return_sequences=True, 
                   input_shape=(5,2)))
model.add(LSTM(100))
model.add(Dense(256))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/keras25_MCP_1_boston.py/'

# model = load_model(load_filepath + '0707_1753_0096-20.8518.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)




end_time = time.time() - start_time

#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict.shape)
print(y_test.shape)
# y_predict = y_predict.reshape(102,13)
# y_test = y_test.reshape(102,)

print(y_test.shape)
print(y_predict.shape)
print(y_test)
print(y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)

# DNN
# loss :  [1966.503662109375, 34.46048355102539]   
# r2스코어 :  0.7021901376455495

# CNN
# loss :  4394.58447265625
# r2스코어 :  0.3228724792763501

# RNN
# loss :  5486.64794921875
# r2스코어 :  0.15460495093469873