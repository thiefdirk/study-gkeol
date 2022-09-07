import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, MaxPool2D, GlobalAveragePooling2D
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
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
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


# trainable = False
# y_predict :  [6 8 1 0 6]
# score :  0.6089
# loss :  1.1579949855804443
# acc :  0.6089000105857849

# trainable = True
# y_predict :  [3 8 8 0 6]
# score :  0.7813
# loss :  2.3832709789276123
# acc :  0.7813000082969666

# GlobalAveragePooling2D
# y_predict :  [3 8 8 0 6]
# score :  0.7918
# loss :  2.484745502471924
# acc :  0.7918000221252441