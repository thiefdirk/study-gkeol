from multiprocessing.dummy import active_children
from re import X
import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
import time
import tensorflow as tf
print(tf.__version__) # 2.9.1

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
activation = 'relu'
drop = 0.2
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.001

# optimizer = adam.Adam(lr=learning_rate) # learning_rate default = 0.001
# optimizer = adadelta.Adadelta(lr=learning_rate)
# optimizer = adagrad.Adagrad(lr=learning_rate)
# optimizer = adamax.Adamax(lr=learning_rate)
# optimizer = rmsprop.RMSprop(lr=learning_rate)
optimizer = nadam.Nadam(lr=learning_rate)

inputs = Input(shape=(28,28,1), name='input')
x = Conv2D(128, (2,2), padding='valid',
           activation=activation, name='Conv2D1')(inputs)
x = Dropout(drop)(x)
x = Conv2D(64, (2,2), padding='same',
           activation=activation, name='Conv2D2')(x)
x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3,3), padding='valid',
           activation=activation, name='Conv2D3')(x)
x = Dropout(drop)(x)
x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
x = Dense(100, activation=activation, name='hidden3')(x)
x = Dropout(drop)(x)
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')


# def create_hyperparameters():
#     batchs = [100, 200, 300, 400, 500]
#     optimizers = ['rmsprop', 'adam', 'adadelta']
#     dropout = [0.3, 0.4, 0.5]
#     activation = ['relu', 'linear', 'selu', 'elu', 'sigmoid']
#     return {'batch_size': batchs, 'optimizer': optimizers, 'drop': dropout, 'activation': activation}

# hyperparameters = create_hyperparameters()
# # print(hyperparameters)

# from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# keras_model =KerasClassifier(build_fn=build_model, verbose=1)

# model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=2, verbose=1)
start = time.time()
hist = model.fit(x_train, y_train, epochs=30, validation_split=0.2, batch_size=400, verbose=1)
end = time.time() - start
loss, acc = model.evaluate(x_test, y_test)

print(learning_rate)

print('acc : ', round(acc, 4))

print('loss : ', round(loss, 4))

print('걸린시간 : ', end)

##################### 시각화 #####################
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

# 1
plt.subplot(2, 1, 1) # 2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) # 2행 1열 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc']) # plt.legend(['acc', 'val_acc'], loc='upper right')

plt.show()