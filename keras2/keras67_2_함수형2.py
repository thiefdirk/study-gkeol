from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16, InceptionV3
from keras.datasets import cifar100
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# model = VGG16()
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) # inceptionV3 minimum input shape is 75x75
# base_model.trainable = False

# inceptionV3.summary()
# print(len(inceptionV3.weights)) # 378
# base_model.trainable = False # 훈련을 동결한다. (가중치를 동결한다.)
# inceptionV3.summary()
# print(len(inceptionV3.weights)) # 376
# exit()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(102, activation='relu')(x)

output1 = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)
#1.
# for layer in base_model.layers:
#     layer.trainable = False
# model.summary()

# Total params: 22,022,082
# Trainable params: 219,298
# Non-trainable params: 21,802,784

#2.
# base_model.trainable = False
# model.summary()

# Total params: 22,022,082
# Trainable params: 219,298
# Non-trainable params: 21,802,784

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