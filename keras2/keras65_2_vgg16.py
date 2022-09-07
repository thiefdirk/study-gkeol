import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.applications import VGG16
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# model = VGG16()
vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))

# vgg16.summary()
vgg16.trainable = False # 훈련을 동결한다. (가중치를 동결한다.)
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=50, factor=0.5, verbose=1)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=500, batch_size=1000, validation_split=0.2, callbacks=[es, lr])

from sklearn.metrics import accuracy_score
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print('y_predict : ', y_predict[:5])
print('score : ', accuracy_score(y_test, y_predict))
print('loss : ', results[0])
print('acc : ', results[1])
