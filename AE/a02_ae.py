import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255. # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape(10000, 784).astype('float32')/255. # (10000, 28, 28) -> (10000, 784)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# model = autoencoder(154) # pca 95%
model = autoencoder(331) # pca 99%

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2)

import matplotlib.pyplot as plt

output = model.predict(x_test)

# 랜덤 숫자 10개 리스트
random_test = np.random.randint(x_test.shape[0], size=5)
print(random_test)

for x,i in enumerate(random_test):
    ax = plt.subplot(2,5,x+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    plt.ylabel('input')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2,5,x+6)
    plt.imshow(output[i].reshape(28,28))
    plt.gray()
    plt.ylabel('output')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout() # 그래프 간격 조정
plt.show()
    
    
