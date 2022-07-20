import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf



#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/train/',
    target_size=(150,150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/test/',
    target_size=(150,150),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) #Found 120 images belonging to 2 classes.

print(xy_train) 
#<keras.preprocessing.image.DirectoryIterator object at 0x000002C22310F9D0>

# print(xy_train[0][0]) # 마지막 배치
print(xy_train[0][0].shape,xy_train[0][1].shape)
# print(xy_train[0][1])
print(xy_test[0][0].shape,xy_test[0][1].shape)

np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1])





# 현재 5,200,200,1 짜리 데이터가 32덩어리



# #2. 모델
# model = Sequential()
# model.add(Conv2D(10,(2,2), input_shape=(100,100,1), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(10,(3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(10,(3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(10,(3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # model.fit(xy_train[0][0], xy_train[0][1]) #배치사이즈를 최대로 잡으면 이거도 건흥
# hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32,
#                     # 전체데이터/batch = 160/5 = 32
#                     validation_data=xy_test,
#                     validation_steps=24) # 생각이 안나심, 알아서 찾으라고 하심

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

# # loss :  0.2754662036895752
# # val_loss :  0.19025221467018127
# # accuracy :  0.8500000238418579
# # val_accuracy :  0.9333333373069763