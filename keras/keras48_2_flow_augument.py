from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_test.shape) # (40000,)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 40000 # 증폭
randindx = np.random.randint(x_train.shape[0], size = augument_size)
print(randindx,randindx.shape) # (40000,)
print(np.max(randindx), np.min(randindx)) # 59997 2
print(type(randindx)) # <class 'numpy.ndarray'>

x_augumented = x_train[randindx].copy()
print(x_augumented,x_augumented.shape) # (40000, 28, 28)
y_augumented = y_train[randindx].copy()
print(y_augumented,y_augumented.shape) # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
# 이과정을 통해 x_augumented 만 변형을 해서 [0] 을붙어 뽑아내준다

print(x_augumented)
print(x_augumented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape)


print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,)
print(np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)
# np.tile(A, repeat_shape), A 배열이 repeat_shape 형태로 반복되어 쌓인 형태가 반환됨

print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) #(100,)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1), # x
    np.zeros(augument_size),                                              # y
    batch_size=augument_size,
    shuffle=True,
).next()
############################# .next() 사용###################################
print(x_data)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001A4AD12D1C0>
print(x_data[0])
print('================================')
# print(x_data[0][0])
print('================================')
# print(x_data[0][1])
print('================================')
print(x_data[0][0].shape) #(28, 28, 1)
print(x_data[0][1].shape) #(28, 28, 1)

############################ .next() 미사용##################################
# print(x_data)
# # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001A4AD12D1C0>
# print(x_data[0][0])
# print('================================')
# # print(x_data[0][0][0])
# print('================================')
# # print(x_data[0][0][1])
# print('================================')
# print(x_data[0][0][0].shape) #(28, 28, 1)
# print(x_data[0][0][1].shape) #(28, 28, 1)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
    # plt.imshow(x_data[0][0][1], cmap='gray') .next() 미사용
plt.show()    

# 데이터를 플로우로 불러오면서 트레인 데이터젠으로 증폭, 증폭횟수는 현재는 augument_size