from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_test.shape) # (40000,)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)


test_datagen = ImageDataGenerator(
    rescale=1./255,

)


augument_size = 10 # 증폭
randindx = np.random.randint(x_train.shape[0], size = augument_size)
print(randindx,randindx.shape) # (40000,)
print(np.max(randindx), np.min(randindx)) # 59997 2
print(type(randindx)) # <class 'numpy.ndarray'>
x_train1 = x_train[randindx].copy()
print(x_train,x_train.shape) # (40000, 28, 28)
x_augumented = x_train[randindx].copy()
print(x_augumented,x_augumented.shape) # (40000, 28, 28)
y_augumented = y_train[randindx].copy()
print(y_augumented,y_augumented.shape) # (40000,)

x_train1 = x_train1.reshape(10, 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], x_augumented.shape[2], 1)
print(x_train1.shape, x_augumented.shape) #(10, 28, 28, 1) (10, 28, 28, 1)
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
# 이과정을 통해 x_augumented 만 변형을 해서 [0] 을붙어 뽑아내준다

print(x_augumented)
print(x_augumented.shape) #(10, 28, 28, 1)

x_train1 = np.concatenate((x_train1, x_augumented))
# y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) # (20, 28, 28, 1) (60000,)

# [실습]
# x_augumented 10개와 x_train 10(subplot)개를 비교하는 이미지 출력


print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,)
print(np.tile(x_train1[0].reshape(28*28),augument_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)
# np.tile(A, repeat_shape), A 배열이 repeat_shape 형태로 반복되어 쌓인 형태가 반환됨

print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) #(100,)

x_data = test_datagen.flow(
    x_train1.reshape(-1,28,28,1), # x
    np.zeros(20),                 # y
    batch_size=20,
    shuffle=False,
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
print(x_data[0][9].shape) #(28, 28, 1)
print(x_data[0][10].shape) #(28, 28, 1)
print(x_data[0][18].shape) #(28, 28, 1)

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
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
    # plt.imshow(x_data[0][0][i], cmap='gray') .next() 미사용
plt.show()    

# plt.subplot()
# plt.imshow(x_data[0][1], cmap='gray')
# plt.show()   

# 데이터를 플로우로 불러오면서 트레인 데이터젠으로 증폭, 증폭횟수는 현재는 augument_size