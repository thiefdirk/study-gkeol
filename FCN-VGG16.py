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

# Vgg16.trainable = False # freeze the Vgg16

def FCN_8s():
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    for layer in model.layers:
        layer.trainable = False
    x = model.output
    x = conv_block(x, 4096, 7, 1)
    x = conv_block(x, 4096, 1, 1)
    x = conv_block(x, 4096, 1, 1)
    block5_conv1 = model.get_layer('block4_pool').output
    block5_conv1 = conv_block(block5_conv1, 256, 1, 1)
    block4_conv1 = model.get_layer('block3_pool').output
    block4_conv1 = conv_block(block4_conv1, 256, 1, 1)
    x = deconv_block(x, 256, 4, 2)
    concat1 = tf.keras.layers.Concatenate()([x, block5_conv1])
    x = deconv_block(concat1, 256, 4, 2)
    concat2 = tf.keras.layers.Concatenate()([x, block4_conv1])
    # 8x upsampling
    x = deconv_block(concat2, 12, 16, 8)
    x = tf.keras.layers.Softmax()(x)
    # x = deconv_block(concat2, 128, 16, 8)
    # x = tf.keras.layers.Conv2D(1, 1, 1, padding='same', activation='softmax')(x)
    model = tf.keras.Model(inputs=model.input, outputs=x)
    return model



# build the model
model = FCN_8s()
# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print the model summary
model.summary()

# train the model
model.fit(train_image_label, epochs=1, validation_data=test_image_label)

# save the model
# model.save('FCN_VGG16.h5')

# load the model
# model = tf.keras.models.load_model('FCN_VGG16.h5')

# predict the image
def predict_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0 # normalization
    image = tf.cast(image, tf.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    predict = model.predict(image)
    predict = np.argmax(predict, axis=3) 
    predict = np.squeeze(predict, axis=0)
    return predict

predict = predict_image('D:\study_data\_data\dataset1\images_prepped_test/0016E5_07959.png')


def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou

image, annotation = next(iter(test_image_label)) # (1, 224, 224, 3), (1, 224, 224, 1), next : get next element
image = image.numpy()
annotation = annotation.numpy()

print(annotation.shape)
print(image.shape)
print(predict.shape)
iou_predict = predict.reshape(256, 256, 1)
score = iou(annotation[0], iou_predict) # 0.0

# get_value from tensor

print('score:', score.numpy())


original_image = image[0]


original_lable = annotation[0]
# predict the image
# show the image
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('original_image')
plt.subplot(1, 3, 2)
plt.imshow(original_lable)
plt.title('original_lable')
plt.subplot(1, 3, 3)
plt.imshow(predict)
plt.title('predict')
plt.show()



