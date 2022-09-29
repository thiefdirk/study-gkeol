import os
import zipfile
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, BatchNormalization, ReLU, Concatenate, Softmax
# import BatchNormalization and ReLU
from tensorflow.keras.layers import BatchNormalization, ReLU
 
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import load_image
import glob
import cv2
import random
from keras.preprocessing.image import ImageDataGenerator

# download the dataset (fcnn-dataset.zip)

# pixel labels in the video frames
class_names = ['sky', 'building','column/pole', 'road', 
               'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void']


train_image_path = 'D:\study_data\_data\dataset1/images_prepped_train/'
train_label_path = 'D:\study_data\_data\dataset1/annotations_prepped_train/'
test_image_path = 'D:\study_data\_data\dataset1/images_prepped_test/'
test_label_path = 'D:\study_data\_data\dataset1/annotations_prepped_test/'
 
BATCH_SIZE = 32

# get the paths to the images
training_image_paths, training_label_map_paths = get_dataset_slice_paths(train_image_path, train_label_path)
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths(test_image_path, test_label_path)
# generate the train and valid sets
training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names))
 
# print class name - normalized RGB tuple pairs
# the tuple values will be multiplied by 255 in the helper functions later
# to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
for class_name, color in zip(class_names, colors):
    print(f'{class_name} -- {color}')

# load the dataset
def load_data(image_path, label_path):
    image_list = []
    label_list = []
    for image in os.listdir(image_path):
        image_list.append(image_path + image)
    for label in os.listdir(label_path):
        label_list.append(label_path + label)
    return image_list, label_list

# load the train dataset
train_image_list, train_label_list = load_data(train_image_path, train_label_path)
# load the test dataset
test_image_list, test_label_list = load_data(test_image_path, test_label_path)

# shuffle the train dataset
train_image_list = tf.random.shuffle(train_image_list)
train_label_list = tf.random.shuffle(train_label_list)


# load the image and label
def load_image_label(image_path, label_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, [256, 256])
    label = tf.cast(label, tf.float32) / 255.0
    return image, label

# load the train image and label
train_image_label = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))
train_image_label = train_image_label.map(load_image_label)
train_image_label = train_image_label.batch(BATCH_SIZE)
# load the test image and label
test_image_label = tf.data.Dataset.from_tensor_slices((test_image_list, test_label_list))
test_image_label = test_image_label.map(load_image_label)
test_image_label = test_image_label.batch(BATCH_SIZE)


def conv_block(inputs, filters, kernel_size, strides, padding='same'): # padding='same' or 'valid'
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def deconv_block(inputs, filters, kernel_size, strides, padding='same'):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x
Vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
Vgg16.trainable = False # freeze the Vgg16

def FCN_8s():
    x = Vgg16(Vgg16.input)
    x = conv_block(x, 4096, 1, 1)
    x = conv_block(x, 4096, 1, 1)
    x = conv_block(x, 4096, 1, 1)
    block5_conv1 = Vgg16.get_layer('block5_conv1').output
    block5_conv1 = conv_block(block5_conv1, 256, 1, 1)
    block4_conv1 = Vgg16.get_layer('block4_conv1').output
    block4_conv1 = conv_block(block4_conv1, 256, 1, 1)
    x = deconv_block(x, 256, 4, 2)
    concat1 = tf.keras.layers.Concatenate()([x, block5_conv1])
    x = deconv_block(concat1, 256, 4, 2)
    concat2 = tf.keras.layers.Concatenate()([x, block4_conv1])
    # 8x upsampling
    x = deconv_block(concat2, 2, 16, 8)
    x = tf.keras.layers.Softmax()(x)
    # x = deconv_block(concat2, 128, 16, 8)
    # x = tf.keras.layers.Conv2D(1, 1, 1, padding='same', activation='softmax')(x)
    return tf.keras.Model(inputs=Vgg16.input, outputs=x)

# Vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
# Vgg16.trainable = False # freeze the Vgg16
# block5_conv1 = Vgg16.get_layer('block5_conv1').output
# block4_conv1 = Vgg16.get_layer('block4_conv1').output


# model1 = Sequential()
# model1.add(Vgg16)
# model1.add(Conv2D(4096, 1, 1, padding='same'))
# model1.add(BatchNormalization())
# model1.add(ReLU())
# model1.add(Conv2D(4096, 1, 1, padding='same'))
# model1.add(BatchNormalization())
# model1.add(ReLU())
# model1.add(Conv2DTranspose(256, 4, 2, padding='same'))
# model1.add(BatchNormalization())
# model1.add(ReLU())
# model1.output = Concatenate()([model1.output, block5_conv1])
# model1.add(Conv2DTranspose(128, 4, 2, padding='same'))
# model1.add(BatchNormalization())
# model1.add(ReLU())
# model1.output = Concatenate()([model1.output, block4_conv1])
# model1.add(Conv2DTranspose(64, 16, 8, padding='same'))
# model1.add(BatchNormalization())
# model1.add(ReLU())
# model1.add(Conv2D(2, 1, 1, padding='same'))
# model1.add(Softmax())


# model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model1.fit(train_image_label, epochs=10, validation_data=test_image_label)

# model1.summary()

# # save the model
# model1.save('FCN_8s.h5')

# # load the model
# # model1 = tf.keras.models.load_model('FCN_8s.h5')

# # predict the test image
# pred = model1.predict(test_image_label)
# pred = np.argmax(pred, axis=-1)
# pred = np.expand_dims(pred, axis=-1)
# pred = pred * 255.0
# pred = pred.astype(np.uint8)

# # show the predict image
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(pred[i])

# plt.show()


fcnn = FCN_8s()
fcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fcnn.fit(train_image_label, epochs=10, validation_data=test_image_label)
fcnn.summary()

# save the model
fcnn.save('FCN_8s.h5')

# load the model
# fcnn = tf.keras.models.load_model('FCN_8s.h5')

# predict the test image
pred = fcnn.predict(test_image_label)
pred = np.argmax(pred, axis=-1)
pred = np.expand_dims(pred, axis=-1)
pred = pred * 255.0
pred = pred.astype(np.uint8)

# show the predict image
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(pred[i])
    
plt.show()


# Model: "vgg16"
# _________________________________________________________________ 
# Layer (type)                Output Shape              Param #       
# ================================================================= 
# input_1 (InputLayer)        [(None, 256, 256, 3)]     0

#  block1_conv1 (Conv2D)       (None, 256, 256, 64)      1792        

#  block1_conv2 (Conv2D)       (None, 256, 256, 64)      36928       

#  block1_pool (MaxPooling2D)  (None, 128, 128, 64)      0

#  block2_conv1 (Conv2D)       (None, 128, 128, 128)     73856       

#  block2_conv2 (Conv2D)       (None, 128, 128, 128)     147584      

#  block2_pool (MaxPooling2D)  (None, 64, 64, 128)       0

#  block3_conv1 (Conv2D)       (None, 64, 64, 256)       295168      

#  block3_conv2 (Conv2D)       (None, 64, 64, 256)       590080      

#  block3_conv3 (Conv2D)       (None, 64, 64, 256)       590080      

#  block3_pool (MaxPooling2D)  (None, 32, 32, 256)       0

#  block4_conv1 (Conv2D)       (None, 32, 32, 512)       1180160     

#  block4_conv2 (Conv2D)       (None, 32, 32, 512)       2359808     

#  block4_conv3 (Conv2D)       (None, 32, 32, 512)       2359808     

#  block4_pool (MaxPooling2D)  (None, 16, 16, 512)       0

#  block5_conv1 (Conv2D)       (None, 16, 16, 512)       2359808     

#  block5_conv2 (Conv2D)       (None, 16, 16, 512)       2359808     

#  block5_conv3 (Conv2D)       (None, 16, 16, 512)       2359808     

#  block5_pool (MaxPooling2D)  (None, 8, 8, 512)         0

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________