from multiprocessing.dummy import active_children
import numpy as np
from keras.datasets import mnist, cifar100
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
print(tf.__version__) # 2.9.1

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

scaler =MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)


from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.01
optimizer = adam.Adam(lr=learning_rate)
activation = 'relu'
drop = 0.2

inputs = Input(shape=(32,32,3), name='input')
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
outputs = Dense(100, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, model = 'auto', verbose=1) # ReduceLROnPlateau: 학습률을 조정해주는 함수

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=1000, verbose=1, callbacks=[reduce_lr, earlyStopping])
end = time.time() - start
result = model.evaluate(x_test, y_test)
print('걸린시간 : ', end)

print('result : ', result)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))

# 걸린시간 :  110.16681671142578
# result :  [3.6302993297576904, 0.14509999752044678]
# accuracy_score :  0.1451

# 걸린시간 :  542.67236328125
# result :  [3.1111459732055664, 0.24279999732971191]
# accuracy_score :  0.2428