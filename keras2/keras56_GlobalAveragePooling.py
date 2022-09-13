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
optimizer = 'adam'

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
model.fit(x_train, y_train, epochs=5, validation_split=0.2, batch_size=400, verbose=1)
end = time.time() - start
result = model.evaluate(x_test, y_test)
print('걸린시간 : ', end)

print('result : ', result)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, np.argmax(y_pred, axis=1)))

# 걸린시간 :  711.6894834041595
# 최적의 매개변수 :  {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 400, 'activation': 'relu'}
# 최고의 점수 :  0.9780833125114441
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CD3BF45670>
# result :  0.982699990272522