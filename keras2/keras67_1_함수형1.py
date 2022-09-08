from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.applications import VGG16
from keras.datasets import cifar100
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# model = VGG16()
vgg16 = VGG16(weights='imagenet', include_top=False,
                input_shape=(32, 32, 3))

# vgg16.summary()
vgg16.trainable = False # 훈련을 동결한다. (가중치를 동결한다.)
# vgg16.summary()

# 함수형 모델
input1 = Input(shape=(32, 32, 3))
vgg16 = VGG16(weights='imagenet', include_top=False)(input1)
dense1 = Dense(100)(vgg16)
output = Dense(100, activation='softmax')(dense1)
model = Model(inputs=input1, outputs=output)


from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=80, mode='min', verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', patience=40, factor=0.5, verbose=1)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2, callbacks=[es, lr])

from sklearn.metrics import accuracy_score
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print('y_predict : ', y_predict[:5])
# print('score : ', accuracy_score(y_test, y_predict))
print('loss : ', results[0])
print('acc : ', results[1])