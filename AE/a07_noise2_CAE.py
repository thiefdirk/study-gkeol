import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose


(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255. # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape(10000, 784).astype('float32')/255. # (10000, 28, 28) -> (10000, 784)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # np.clip는 최소값과 최대값을 지정해주는 함수입니다.
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

x_train_noised = x_train_noised.reshape(60000, 28, 28, 1)
x_test_noised = x_test_noised.reshape(10000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

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


model = autoencoder(154) # pca 95%
# model = autoencoder(331) # pca 99%

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10, batch_size=256, validation_split=0.2)

import matplotlib.pyplot as plt

output = model.predict(x_test_noised)

# 랜덤 숫자 10개 리스트
random_test = np.random.randint(x_test.shape[0], size=5)

for x,i in enumerate(random_test):
    ax = plt.subplot(3,5,x+1)
    if x == 0:
        ax.set_ylabel('original', size=20)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3,5,x+6)
    if x == 0:
        ax.set_ylabel('noised', size=20)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3,5,x+11)
    if x == 0:
        ax.set_ylabel('ouput', size=20)
    plt.imshow(output[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout() # 그래프 간격 조정
plt.show()
    
    
