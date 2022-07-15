from sklearn.metrics import r2_score, accuracy_score
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################
import time
start = time.time()


#1. 데이터

import numpy as np
x1_datasets = np.array([range(100), range(301,401)]) # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]) # 원유, 돈육, 밀
x1 = np.transpose(x1_datasets) 
x2 = np.transpose(x2_datasets)
print(x1.shape, x2.shape) # (100, 2) (100, 3)

y = np.array(range(2001,2101)) # 금리 (100, )

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(x1_train.shape, x1_test.shape) # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape) # (80, 3) (20, 3)
print(y_train.shape, y_test.shape) # (80,) (20,)

#2. 모델
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(100, activation='relu', name='ak1')(input1)
dense2 = Dense(200, activation='relu', name='ak2')(dense1)
dense3 = Dense(300, activation='relu', name='ak3')(dense2)
output1 = Dense(100, activation='relu', name='out_ak1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(1100, activation='relu', name='ak11')(input2)
dense12 = Dense(120, activation='relu', name='ak12')(dense11)
dense13 = Dense(130, activation='relu', name='ak13')(dense12)
dense14 = Dense(140, activation='relu', name='ak14')(dense13)
output2 = Dense(100, activation='relu', name='out_ak2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate # 앙상블모델
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(20, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

print(merge1.shape)

model = Model(inputs=[input1, input2], outputs=last_output)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
model.summary()
loss = model.evaluate([x1_test, x2_test], y_test)
print(x1_test.shape, x2_test.shape)
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('loss : ', loss)
print('r2스코어 : ', r2)
print("time :", time.time() - start)

# loss :  [0.0003935694694519043, 0.00933837890625]
# r2스코어 :  0.9999995020172404
# time : 41.14100527763367
