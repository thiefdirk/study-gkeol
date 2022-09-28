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


# define the model
def FCN_VGG16():
    # load the VGG16 model
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    # freeze the layers
    for layer in model.layers:
        layer.trainable = False
    # add the FCN layers
    x = model.output
    x = tf.keras.layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(12, (1, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(12, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(12, (1, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(12, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(12, (1, 1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(12, (16, 16), strides=(8, 8), padding='same')(x)
    x = tf.keras.layers.Activation('softmax')(x)
    # build the model
    model = tf.keras.Model(inputs=model.input, outputs=x)
    return model


# build the model
model = FCN_VGG16()
# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print the model summary
model.summary()

# train the model
model.fit(train_image_label, epochs=10, validation_data=test_image_label)

# save the model
model.save('FCN_VGG16.h5')

# load the model
model = tf.keras.models.load_model('FCN_VGG16.h5')

# predict the image
def predict_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    predict = model.predict(image)
    predict = np.argmax(predict, axis=-1)
    predict = np.squeeze(predict, axis=0)
    return predict

# predict the image
predict = predict_image('D:\study_data\_data\dataset1\images_prepped_test/0016E5_07959.png')
# show the image
plt.imshow(predict)
plt.show()

