import numpy as np
from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import time
import tensorflow as tf
print(tf.__version__) # 2.9.1

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam

# learning_rate = 0.001

# # optimizer = adam.Adam(lr=learning_rate) # learning_rate default = 0.001
# # optimizer = adadelta.Adadelta(lr=learning_rate)
# # optimizer = adagrad.Adagrad(lr=learning_rate)
# # optimizer = adamax.Adamax(lr=learning_rate)
# # optimizer = rmsprop.RMSprop(lr=learning_rate)
# optimizer = nadam.Nadam(lr=learning_rate)

#2. 모델
def build_model(drop=0.5, optimizer=adam.Adam, activation = 'relu', learning_rate=0.001, node1=512, node2=256, node3=128):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer(lr=learning_rate), metrics=['acc'], loss='sparse_categorical_crossentropy')
    
    return model

def create_hyperparameters():
    batchs = [100, 200, 300, 400, 500]
    optimizers = [rmsprop.RMSprop, adam.Adam, adadelta.Adadelta]
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'selu', 'elu', 'sigmoid']
    learning_rate = [0.001, 0.01, 0.1]
    return {'batch_size': batchs, 'optimizer': optimizers, 'drop': dropout, 'activation': activation, 'learning_rate': learning_rate}

hyperparameters = create_hyperparameters()
# print(hyperparameters)

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

keras_model =KerasClassifier(build_fn=build_model, verbose=1)

model = GridSearchCV(keras_model, hyperparameters, cv=2, verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
end = time.time() - start
result = model.score(x_test, y_test)
print('걸린시간 : ', end)
print('최적의 매개변수 : ', model.best_params_)
print('최고의 점수 : ', model.best_score_)
print('model.best_estimator_ : ', model.best_estimator_)
print('result : ', result)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))

# 걸린시간 :  711.6894834041595
# 최적의 매개변수 :  {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 400, 'activation': 'relu'}
# 최고의 점수 :  0.9780833125114441
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001CD3BF45670>
# result :  0.982699990272522

# 걸린시간 :  5266.268299102783
# 최적의 매개변수 :  {'activation': 'relu', 'batch_size': 100, 'drop': 0.3, 'learning_rate': 0.001, 'optimizer': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>}
# 최고의 점수 :  0.9699166715145111
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001DDDABF6730>
# result :  0.974399983882904
# accuracy_score :  0.9744