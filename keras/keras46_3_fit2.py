import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf


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
    'd:/study_data/_data/image/brain/train/',
    target_size=(100,100),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/test/',
    target_size=(100,100),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) #Found 120 images belonging to 2 classes.

print(xy_train) 
#<keras.preprocessing.image.DirectoryIterator object at 0x000002C22310F9D0>

print(xy_train[0][0]) # 마지막 배치
print(xy_train[0][0].shape)
print(xy_train[0][1])
print(xy_train[0][1].shape)

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

# 현재 5,200,200,1 짜리 데이터가 32덩어리

#2. 모델
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(100,100,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(xy_train[0][0], xy_train[0][1]) #배치사이즈를 최대로 잡으면 이거도 건흥
# hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32,
#                     # 전체데이터/batch = 160/5 = 32
#                     validation_data=xy_test,
#                     validation_steps=24) # 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다.

####################fit_gen 대신에 fit써도됨###################################
hist = model.fit(xy_train, epochs=30, steps_per_epoch=32,
                    # 전체데이터/batch = 160/5 = 32
                    validation_data=xy_test,
                    validation_steps=24) 
#############################################################################

hist = model.fit(xy_train, epochs=30, steps_per_epoch=32,
                    # 전체데이터/batch = 160/5 = 32
                    # validation_data=xy_test,
                    # validation_steps=24) 
                    validation_split=0.2)


acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_accuracy : ', val_accuracy[-1])

import matplotlib.pyplot as plt
# plt.imshow(acc, 'gray')
plt.plot(acc, 'gray')
plt.show()

# loss :  0.29268208146095276
# val_loss :  0.1773398071527481
# accuracy :  0.875
# val_accuracy :  0.9416666626930237