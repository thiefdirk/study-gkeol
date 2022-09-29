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
 
BATCH_SIZE = 16



def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):
    '''
    Preprocesses the dataset by:
        * resizing the input image and label maps
        * normalizing the input image pixels
        * reshaping the label maps from (height, width, 1) to (height, width, 12)
    
    Args:
        t_filename(string) -- path to the raw input image
        a_filename(string) -- path to the raw annotation (label map) file
        height(int) -- height in pixels to resize to
        width(int) -- width in pixels to resize to
    
    Returns:
        image(tensor) -- preprossed image
        annotation(tensor) -- preprocessed annotation
    '''
 
    # Convert image and mask files to tensors
    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)
 
    # Resize image and segmentation mask
    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []
 
    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:,:,0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))
    
    annotation = tf.stack(stack_list, axis=2)
 
    # Normalize pixels in the input image
    image = image / 127.5
    image -= 1
 
    return image, annotation
 
def get_dataset_slice_paths(image_dir, label_map_dir):
    '''
    generates the lists of image and label map paths
  
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory
 
    Returns:
        image_paths (list of strings) -- paths to each image file
        label_map_paths (list of strings) -- paths to each label map
    '''
 
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]
 
    return image_paths, label_map_paths
 
def get_training_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the training set.
  
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory
 
    Returns:
        tf Dataset containing the preprocessed train set
    '''
    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)
 
    return training_dataset
 
def get_validation_dataset(image_paths, label_map_paths):
    '''
    Prepares shuffled batches of the validation set.
  
    Args:
        image_dir (string) -- path to the input images directory
        label_map_dir (string) -- path to the label map directory
 
    Returns:
        tf Dataset containing the preprocessed train set
    '''
    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()
 
    return validation_dataset


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
Vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
Vgg16.trainable = False # freeze the Vgg16

def FCN_8s():
    x = Vgg16(Vgg16.input)
    x = conv_block(x, 4096, 7, 1)
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
    x = deconv_block(concat2, 12, 16, 8)
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

train_count = len(training_image_paths)
validation_count = len(validation_image_paths)

steps_per_epoch = train_count // BATCH_SIZE
validation_steps = validation_count // BATCH_SIZE


# fcnn = FCN_8s()
# fcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fcnn.fit(training_dataset, epochs=100, validation_data=validation_dataset, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
# fcnn.summary()

# # save the model
# fcnn.save('FCN_8s.h5')

# load the model
fcnn = tf.keras.models.load_model('FCN_8s.h5')

# predict the test image
pred = fcnn.predict(validation_dataset, steps=validation_steps)
pred = np.argmax(pred, axis=3) # (batch_size, 224, 224)

# iou
def show_metrics(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou


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