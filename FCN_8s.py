import os
import zipfile
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
 
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

Vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
# download the dataset (fcnn-dataset.zip)

# pixel labels in the video frames
class_names = ['sky', 'building','column/pole', 'road', 
               'side walk', 'vegetation', 'traffic light', 'fence', 'vehicle', 'pedestrian', 'bicyclist', 'void']


train_image_path = 'D:\study_data\_data\dataset1/images_prepped_train/'
train_label_path = 'D:\study_data\_data\dataset1/annotations_prepped_train/'
test_image_path = 'D:\study_data\_data\dataset1/images_prepped_test/'
test_label_path = 'D:\study_data\_data\dataset1/annotations_prepped_test/'
 
BATCH_SIZE = 16

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

# def FCN_8s():
#     inputs = tf.keras.layers.Input(shape=(256, 256, 3))
#     Vgg16.trainable = False
#     x = Vgg16(inputs)
#     # encoder
#     x = conv_block(inputs, 64, 3, 1)
#     x = conv_block(x, 64, 3, 1)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = conv_block(x, 128, 3, 1)
#     x = conv_block(x, 128, 3, 1)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = conv_block(x, 256, 3, 1)
#     x = conv_block(x, 256, 3, 1)
#     x = conv_block(x, 256, 3, 1)
#     x_28 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = conv_block(x_28, 512, 3, 1)
#     x = conv_block(x, 512, 3, 1)
#     x = conv_block(x, 512, 3, 1)
#     x_14 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     x = conv_block(x_14, 512, 3, 1)
#     x = conv_block(x, 512, 3, 1)
#     x = conv_block(x, 512, 3, 1)
#     x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#     # decoder
#     x_28 = conv_block(x_28, 512, 1, 1)
#     x_14 = conv_block(x_14, 512, 1, 1)
#     x_28 = deconv_block(x_28, 512, 3, 1)
#     x_14 = deconv_block(x_14, 512, 3, 1)
#     x = deconv_block(x, 512, 3, 1)
#     x = tf.keras.layers.Concatenate()([x, x_14])
#     x = deconv_block(x, 512, 3, 1)
#     x = deconv_block(x, 256, 3, 1)
#     x = tf.keras.layers.Concatenate()([x, x_28])
#     x = deconv_block(x, 256, 3, 1)
#     x = deconv_block(x, 128, 3, 1)
#     x = deconv_block(x, 64, 3, 1)
#     x = deconv_block(x, 64, 3, 1)
#     x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(x)
#     x = tf.keras.layers.Activation('sigmoid')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=x)
#     return model


def FCN_8s():
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    Vgg16.trainable = False
    x = Vgg16(inputs)
    x = conv_block(x, 64, 1, 1)
    x = conv_block(x, 64, 1, 1)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 128, 1, 1)
    x = conv_block(x, 128, 1, 1)
    

# fcn8s
fcn8s = FCN_8s()
# compile the model
fcn8s.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# train the model
fcn8s.fit(train_image_label, epochs=10, validation_data=test_image_label)

# save the model
fcn8s.save('fcn8s.h5')

# load the model
# fcn8s = tf.keras.models.load_model('fcn8s.h5')

# test the model
test_image = tf.io.read_file(test_image_list[0])
test_image = tf.image.decode_png(test_image, channels=3)
test_image = tf.image.resize(test_image, [256, 256])
test_image = tf.cast(test_image, tf.float32) / 255.0
test_image = tf.expand_dims(test_image, axis=0)
test_image = fcn8s.predict(test_image)
test_image = tf.squeeze(test_image, axis=0)
test_image = tf.cast(test_image > 0.5, tf.float32)
test_image = tf.image.resize(test_image, [512, 512])
test_image = tf.squeeze(test_image, axis=2)
plt.imshow(test_image, cmap='gray')
plt.show()

    