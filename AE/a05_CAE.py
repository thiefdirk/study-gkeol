# [실습] 4번 카피 복붙
# CNN으로 바꾸시오
# UpSampling2D

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()


# x_train = x_train.reshape(60000, 784).astype('float32')/255. # (60000, 28, 28) -> (60000, 784)
# x_test = x_test.reshape(10000, 784).astype('float32')/255. # (10000, 28, 28) -> (10000, 784)

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))) # (28, 28, 1) -> (28, 28, 1)
#     model.add(MaxPooling2D((2, 2), padding='same'))# (28, 28, 1) -> (14, 14, 1), same: 출력 크기가 입력 크기와 같도록 패딩
#     model.add(Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same')) # (14, 14, 1) -> (14, 14, 1)
#     model.add(MaxPooling2D((2, 2), padding='same')) # (14, 14, 1) -> (7, 7, 1)
#     model.add(Conv2D(hidden_layer_size/4, (3, 3), activation='relu', padding='same')) # (7, 7, 1) -> (7, 7, 1)
#     model.add(UpSampling2D((2, 2))) # (7, 7, 1) -> (14, 14, 1)
#     model.add(Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same')) # (14, 14, 1) -> (14, 14, 1)
#     model.add(UpSampling2D((2, 2))) # (14, 14, 1) -> (28, 28, 1)
#     model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same')) # (28, 28, 1) -> (28, 28, 1)
#     return model

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))) # (28, 28, 1) -> (28, 28, 1)
    model.add(MaxPooling2D((2, 2), padding='same'))# (28, 28, 1) -> (14, 14, 1)
    model.add(Conv2D(hidden_layer_size/2, (3, 3), activation='relu', padding='same')) # (14, 14, 1) -> (14, 14, 1)
    model.add(MaxPooling2D((2, 2), padding='same')) # (14, 14, 1) -> (7, 7, 1)
    model.add(Conv2D(hidden_layer_size/4, (3, 3), activation='relu', padding='same')) # (7, 7, 1) -> (7, 7, 1)
    model.add(Conv2DTranspose(hidden_layer_size/2, (7, 7), activation='relu', padding='valid'))
    model.add(Conv2DTranspose(hidden_layer_size, (7, 7), activation='sigmoid', padding='valid'))
    model.add(Conv2DTranspose(1, (10, 10), activation='sigmoid', padding='valid'))
    model.summary()
    return model

model = autoencoder(256) # pca 95%

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train, x_train, epochs=2, batch_size=256, validation_split=0.2)
output = model.predict(x_test)

# 랜덤 숫자 10개 리스트
random_test = np.random.randint(x_test.shape[0], size=5)
print(random_test)

for x,i in enumerate(random_test):
    ax = plt.subplot(2,5,x+1)
    plt.imshow(x_test[i])
    plt.gray()
    plt.ylabel('input')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2,5,x+6)
    plt.imshow(output[i])
    plt.gray()
    plt.ylabel('output')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout() # 그래프 간격 조정
plt.show()
    