import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
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
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

men_women = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/men_women/data/',
    target_size=(100,100),
    batch_size=3309,
    class_mode='categorical',
    shuffle=True
) #Found 3309 images belonging to 2 classes.

test_set = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/men_women/test_set/',
    target_size=(100,100),
    batch_size=1,
    class_mode='categorical',
    shuffle=True
) #Found 1 images belonging to 1 classes.



# print(human) 
#<keras.preprocessing.image.DirectoryIterator object at 0x000002C22310F9D0>

# print(horse[0][0]) # 마지막 배치
# print(horse[0][0].shape)
# print(horse[0][1])
# print(horse[0][1].shape)
# print(human[0][1])
# print(human[0][1].shape)

# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

# [1][0] 남자
# [0][1] 여자

x = men_women[0][0]
y = men_women[0][1]

test_set_x = test_set[0][0]
print(x) 
print(y) 
print(test_set) 


print(x.shape,y.shape) #(840, 100, 100, 3) (840, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,
                                                    shuffle=True
                                                    )



np.save('d:/study_data/_save/_npy/keras47_4_train_x.npy', arr=x_train)
np.save('d:/study_data/_save/_npy/keras47_4_train_y.npy', arr=y_train)
np.save('d:/study_data/_save/_npy/keras47_4_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras47_4_test_y.npy', arr=y_test)
np.save('d:/study_data/_save/_npy/keras47_4_test_set.npy', arr=test_set_x)

