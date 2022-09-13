import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255. # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape(10000, 784).astype('float32')/255. # (10000, 28, 28) -> (10000, 784)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=hidden_layer_size, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# model_01 = autoencoder(1)
# model_04 = autoencoder(4)
# model_16 = autoencoder(16)
# model_32 = autoencoder(32)
# model_64 = autoencoder(64)
# model_154 = autoencoder(154)

# print('============ node 1 ============')
# model_01.compile(optimizer='adam', loss='binary_crossentropy')
# model_01.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)

# print('============ node 4 ============')
# model_04.compile(optimizer='adam', loss='binary_crossentropy')
# model_04.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)

# print('============ node 16 ============')
# model_16.compile(optimizer='adam', loss='binary_crossentropy')
# model_16.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)

# print('============ node 32 ============')
# model_32.compile(optimizer='adam', loss='binary_crossentropy')
# model_32.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)

# print('============ node 64 ============')
# model_64.compile(optimizer='adam', loss='binary_crossentropy')
# model_64.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)

# print('============ node 154 ============')
# model_154.compile(optimizer='adam', loss='binary_crossentropy')
# model_154.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)

node_num = [1, 4, 16, 32, 64, 154]

for p, i in enumerate(node_num):
    print('============ node {} ============'.format(i))
    model = autoencoder(i)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train, x_train, epochs=10, batch_size=256, validation_split=0.2)
    output = model.predict(x_test)
    rand_pick = np.random.randint(output.shape[0], size=5) # 0~9999 중 5개 랜덤 추출
    for j, k in enumerate(rand_pick):
        plt.subplot(12, 5, (p*10)+(j+1))
        plt.imshow(x_test[k].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(12, 5, ((p*10)+5)+(j+1))
        plt.imshow(output[k].reshape(28, 28), cmap='gray')
        plt.axis('off')
plt.tight_layout() # 그래프 간격 조정
plt.show()