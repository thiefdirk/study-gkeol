import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255. # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape(10000, 784).astype('float32')/255. # (10000, 28, 28) -> (10000, 784)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # np.clip는 최소값과 최대값을 지정해주는 함수입니다.
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(154) # pca 95%
# model = autoencoder(331) # pca 99%

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=20, batch_size=256, validation_split=0.2)

import matplotlib.pyplot as plt

output = model.predict(x_test_noised)

# 랜덤 숫자 10개 리스트
random_test = np.random.randint(x_test.shape[0], size=5)
print(random_test)

for x,i in enumerate(random_test):
    ax = plt.subplot(3,6,x+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    plt.ylabel('input')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3,6,x+6)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    plt.ylabel('input')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3,6,x+11)
    plt.imshow(output[i].reshape(28,28))
    plt.gray()
    plt.ylabel('output')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout() # 그래프 간격 조정
plt.show()
    
    
