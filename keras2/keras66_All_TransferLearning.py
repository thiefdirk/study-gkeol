from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNetRS101, ResNetRS152
from keras.applications import DenseNet121, DenseNet169, DenseNet201
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
from keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

model_list = [VGG16, VGG19, ResNet50, ResNet50V2, ResNetRS101, ResNet101V2, 
              ResNetRS152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, 
              InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, MobileNetV3Large, 
              MobileNetV3Small, NASNetLarge, NASNetMobile, EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]

# for model in model_list:
    # models = model(include_top=False,
    #           input_shape=(32, 32, 3))
    # models.trainable = False
    # # model.summary()
    # # print pretrained model's name

    # model = Sequential()
    # model.add(models)
    # # model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    # model.add(Dense(100))
    # model.add(Dropout(0.2))
    # model.add(Dense(100))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='softmax'))
    # # model.summary()
    
    # es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=0)
    # lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, verbose=0)

    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # model.fit(x_train, y_train, epochs=500, batch_size=1000, validation_split=0.2, callbacks=[es, lr], verbose=0)

    # from sklearn.metrics import accuracy_score
    # results = model.evaluate(x_test, y_test)
    # y_predict = model.predict(x_test)
    # y_predict = np.argmax(y_predict, axis=1)
    # print(models.name) 
    # print(len(models.weights)) # 32
    # print(len(models.trainable_weights)) # 32  
    # print('loss : ', results[0])
    # print('acc : ', results[1])
        
    # print('=====================================')
    
    
for model in model_list:
    model = model()
    print(model.name) 
    print('trainable=true')
    print(len(model.weights)) # 32
    print(len(model.trainable_weights)) # 32  
    model.trainable = False
    print('trainable=false')
    print(len(model.weights)) # 32
    print(len(model.trainable_weights)) # 32          
    print('=====================================')

# vgg16
# trainable=true
# 32
# 32
# trainable=false
# 32
# 0
# =====================================
# vgg19
# trainable=true
# 38
# 38
# trainable=false
# 38
# 0
# =====================================
# resnet50
# trainable=true
# 320
# 214
# trainable=false
# 320
# 0
# =====================================
# resnet50v2
# trainable=true
# 272
# 174
# trainable=false
# 272
# 0
# =====================================
# resnet-rs-101
# trainable=true
# 669
# 455
# trainable=false
# 669
# 0
# =====================================
# resnet101v2
# trainable=true
# 544
# 344
# trainable=false
# 544
# 0
# =====================================
# resnet-rs-152
# trainable=true
# 992
# 676
# trainable=false
# 992
# 0
# =====================================
# resnet152v2
# trainable=true
# 816
# 514
# trainable=false
# 816
# 0
# =====================================
# densenet121
# trainable=true
# 606
# 364
# trainable=false
# 606
# 0
# =====================================
# densenet169
# trainable=true
# 846
# 508
# trainable=false
# 846
# 0
# =====================================
# densenet201
# trainable=true
# 1006
# 604
# trainable=false
# 1006
# 0
# =====================================
# inception_v3
# trainable=true
# 378
# 190
# trainable=false
# 378
# 0
# =====================================
# inception_resnet_v2
# trainable=true
# 898
# 490
# trainable=false
# 898
# 0
# =====================================
# mobilenet_1.00_224
# trainable=true
# 137
# 83
# trainable=false
# 137
# 0
# =====================================
# mobilenetv2_1.00_224
# trainable=true
# 262
# 158
# trainable=false
# 262
# 0
# =====================================
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# MobilenetV3large
# trainable=true
# 266
# 174
# trainable=false
# 266
# 0
# =====================================
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.
# MobilenetV3small
# trainable=true
# 210
# 142
# trainable=false
# 210
# 0
# =====================================
# NASNet
# trainable=true
# 1546
# 1018
# trainable=false
# 1546
# 0
# =====================================
# NASNet
# trainable=true
# 1126
# 742
# trainable=false
# 1126
# 0
# =====================================
# efficientnetb0
# trainable=true
# 314
# 213
# trainable=false
# 314
# 0
# =====================================
# efficientnetb1
# trainable=true
# 442
# 301
# trainable=false
# 442
# 0
# =====================================
# efficientnetb7
# trainable=true
# 1040
# 711
# trainable=false
# 1040
# 0
# =====================================
# xception
# trainable=true
# 236
# 156
# trainable=false
# 236
# 0
# =====================================