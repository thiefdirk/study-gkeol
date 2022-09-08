from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2, ResNet101
from keras.applications import ResNetRS101, ResNetRS152
from keras.applications import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.applications import MobileNet, MobileNetV2
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from keras.applications import Xception
from keras.applications import ResNet101V2, ResNet152V2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
from keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

(x_train, y_train), (x_test, y_test) = cifar100.load_data()


model_list = [VGG19,
              ResNet50, ResNet101,DenseNet121, MobileNetV2, EfficientNetB0]
# 에러나도 계속 돌아가는 if문

# for model in model_list:
#     try:
#         model = model(weights='imagenet', include_top=False,
#                       input_shape=(32, 32, 3))
#         # model.trainable = False

#         print('============================================')
#     except:
#         print('============================================')
#         print('Error')
#         print(model.__name__)
#         print('============================================')
#         continue

# ============================================
# Error
# Xception
# ============================================
# ============================================
# Error
# InceptionV3
# ============================================
# ============================================
# Error
# InceptionResNetV2
# ============================================
# ============================================
# Error
# NASNetMobile
# ============================================

for model in model_list:
    base_model = model(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # model.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='softmax'))
    es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=0)
    lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, verbose=0)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, epochs=500, batch_size=1000, validation_split=0.2, callbacks=[es, lr], verbose=0)
    results = model.evaluate(x_test, y_test)
    print('loss : ', results[0])
    print('acc : ', results[1])

# model.trainable = True
# loss :  9.725977897644043
# acc :  0.2687000036239624

# loss :  3.5556678771972656
# acc :  0.5368000268936157

# loss :  7.196557998657227
# acc :  0.37290000915527344

# loss :  2.81032133102417
# acc :  0.6208000183105469

# loss :  4.9280595779418945
# acc :  0.5483999848365784

# loss :  3.768545389175415
# acc :  0.5544999837875366

# model.trainable = False




















