import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.reshape(60000, 784).astype('float32')/255. # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape(10000, 784).astype('float32')/255. # (10000, 28, 28) -> (10000, 784)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(1064, activation='relu')(input_img) 
# encoded = Dense(16, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
autoencoder = Model(input_img, decoded) # 입력값과 출력값을 지정해준다.
# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2)

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

# 랜덤 숫자 10개 리스트
random_test = np.random.randint(x_test.shape[0], size=10)
print(random_test)



n = 10 # 몇 개를 출력할 것인지
plt.figure(figsize=(20, 4)) # 가로 20인치, 세로 4인치
for i in range(n):
    # 원본 이미지를 맨 위에 그린다.
    ax = plt.subplot(2, n, i+1) # 2행 n열의 그림 중 i+1번째 그림
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False) # x축 숫자 제거
    ax.get_yaxis().set_visible(False) # y축 숫자 제거

    # 재구성한 이미지를 아래에 그린다.
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
