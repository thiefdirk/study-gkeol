from dataclasses import dataclass
from multiprocessing.dummy import active_children
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris, load_wine, fetch_covtype
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import train_test_split
from sklearn.model_selection import train_test_split
print(tf.__version__) # 2.9.1

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. 모델
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

learning_rate = 0.01
optimizer = nadam.Nadam(lr=learning_rate)
activation = 'relu'
drop = 0.2


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


# model.add(Dense(256))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(7, activation='softmax'))
# model.summary()

#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.5, model = 'auto', verbose=1) # ReduceLROnPlateau: 학습률을 조정해주는 함수
tb = TensorBoard(log_dir='D:\study_data/tensorboard.log\_graph', histogram_freq=0, write_graph=True, write_images=True)

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=500, verbose=1, callbacks=[reduce_lr, earlyStopping, tb])
end = time.time() - start
result = model.evaluate(x_test, y_test)
print('걸린시간 : ', end)

print('result : ', result)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))
# 걸린시간 :  461.4713611602783
# result :  [0.06379805505275726, 0.9819999933242798]    
# accuracy_score :  0.982