import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score
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

x_train = np.load('d:/study_data/_save/_npy/keras47_3_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_3_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_3_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_3_test_y.npy')

print(x_train.shape,y_train.shape) #(368, 100, 100, 3) (368,)
print(x_test.shape,y_test.shape) #(159, 100, 100, 3) (159,)

#2. 모델
model = Sequential()
model.add(Conv2D(150,(2,2), input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(160, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model('./_test/0718_2055_0031-303197376.0000.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )


hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2,
                 callbacks=[earlyStopping])

acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_accuracy : ', val_accuracy[-1])


# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

# acc = accuracy_score(y_test, y_predict)
print('loss: ', loss)
print('acc스코어 : ', acc[-1])
# print('내일 종가 : ', y_predict)
print("time :", time.time() - start)

# loss:  [0.013227352872490883, 0.9960317611694336]
# acc스코어 :  1.0
# time : 38.17850875854492

import matplotlib.pyplot as plt
# plt.imshow(acc, 'gray')
plt.plot(acc, 'gray')
plt.show()

# import image

# img = image.load_img('d:/study_data/_save/_npy/keras47_3_test_x.npy')