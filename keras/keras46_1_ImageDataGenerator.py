import numpy as np
from keras.preprocessing.image import ImageDataGenerator


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
    'd:/_data/image/brain/train/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
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