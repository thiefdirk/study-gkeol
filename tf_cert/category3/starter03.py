# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(rescale=1/255)
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)

    validation_datagen = ImageDataGenerator(rescale=1/255)
        #Your Code here)

    train_generator = train_datagen.flow_from_directory(
        'tmp/horse-or-human/',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary')
        #Your Code Here)

    validation_generator = validation_datagen.flow_from_directory(
        'tmp/testdata/',
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary')
        #Your Code Here)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])



    model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        #Your Code Here#)

    model.fit_generator(train_generator,
                        steps_per_epoch=8,
                        epochs=1,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=8)
    
    # print accuracy
    from sklearn.metrics import accuracy_score
    y_pred = model.predict_generator(validation_generator)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print(accuracy_score(validation_generator.classes, y_pred))
        
    
    return model
    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")