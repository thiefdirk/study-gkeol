import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.applications import VGG16

# model = VGG16()
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# include_top=False : FC 레이어를 제외한다., weights='imagenet' : ImageNet 가중치를 불러온다.
# FC 레이어 : Fully Connected Layer
# Fully Connected Layer : Dense 레이어
model.summary()

print(len(model.weights)) # 32
print(len(model.trainable_weights)) # 32

########################## include_top=True ##########################
# 1. FC 레이어 원래 그대로
# 2. input_shape=(224, 224, 3) 고정값, 바꿀수없음

#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

#  flatten (Flatten)           (None, 25088) 0
#  fc1 (Dense)                 (None, 4096) 102764544
#  fc2 (Dense)                 (None, 4096) 16781312
#  predictions (Dense)         (None, 1000) 4097000

# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

########################## include_top=False ##########################
# 1. FC 레이어 제거
# 2. input_shape=(224, 224, 3) -> (32, 32, 3)으로 변경 

# input_1 (InputLayer)        [(None, 32, 32, 3)]       0
# block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# .........................................
# .........................................플래튼 포함 하단 제거됨

# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0