import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf



#1. 데이터
# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0])
# np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1])
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0])
# np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1])

x_train = np.load('d:/study_data/_save/_npy/keras46_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras46_5_test_y.npy')

# print(xy_train[0][0]) # 마지막 배치
print(x_train.shape,y_train.shape)
# print(xy_train[0][1])
print(x_test.shape,y_test.shape)

# 현재 5,200,200,1 짜리 데이터가 32덩어리

