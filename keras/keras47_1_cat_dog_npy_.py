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
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/training_set/',
    target_size=(100,100),
    batch_size=8005,
    class_mode='categorical',
    shuffle=True
) #Found 8005 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test_set/',
    target_size=(100,100),
    batch_size=2023,
    class_mode='categorical',
    shuffle=True
) #Found 2023 images belonging to 2 classes.

print(xy_train) 
#<keras.preprocessing.image.DirectoryIterator object at 0x000002C22310F9D0>

print(xy_train[0][0]) # 마지막 배치
print(xy_train[0][0].shape)
print(xy_train[0][1])
print(xy_train[0][1].shape)
print(xy_test[0][1])
print(xy_test[0][1].shape)

# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

np.save('d:/study_data/_save/_npy/keras47_1_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras47_1_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras47_1_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras47_1_test_y.npy', arr=xy_test[0][1])




# # 현재 5,200,200,1 짜리 데이터가 32덩어리

# #2. 모델
# model = Sequential()
# model.add(Conv2D(100,(2,2), input_shape=(100,100,1), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(100,(3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(100,(3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(100,(3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # model.fit(xy_train[0][0], xy_train[0][1]) #배치사이즈를 최대로 잡으면 이거도 건흥
# # hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32,
# #                     # 전체데이터/batch = 160/5 = 32
# #                     validation_data=xy_test,
# #                     validation_steps=24) # 생각이 안나심, 알아서 찾으라고 하심

# hist = model.fit(xy_train[0][0], xy_train[0][1], epochs=30, batch_size=32, validation_split=0.2)

# acc = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', acc[-1])
# print('val_accuracy : ', val_accuracy[-1])

# import matplotlib.pyplot as plt
# # plt.imshow(acc, 'gray')
# plt.plot(acc, 'gray')
# plt.show()

# # loss :  0.10805143415927887
# # val_loss :  0.6262131333351135
# # accuracy :  0.984375
# # val_accuracy :  0.8125

