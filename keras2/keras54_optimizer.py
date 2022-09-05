import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련

from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax, rmsprop, nadam
# from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
# from keras.optimizers.optimizer_v2 import adam

learning_rate = 0.01

# optimizer = adam.Adam(lr=learning_rate)
# optimizer = adadelta.Adadelta(lr=learning_rate)
# optimizer = adagrad.Adagrad(lr=learning_rate)
# optimizer = adamax.Adamax(lr=learning_rate)
# optimizer = rmsprop.RMSprop(lr=learning_rate)
# optimizer = nadam.Nadam(lr=learning_rate)

optimizer_list = [adam.Adam(lr=learning_rate), adadelta.Adadelta(lr=learning_rate), adagrad.Adagrad(lr=learning_rate), adamax.Adamax(lr=learning_rate), rmsprop.RMSprop(lr=learning_rate), nadam.Nadam(lr=learning_rate)]

for optimizer in optimizer_list:
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x, y, epochs=50, batch_size=1, verbose=0)
    loss = model.evaluate(x, y)
    y_predict = model.predict([11])
    print('optimizer : ', optimizer)
    print('loss : ', loss, 'y_predict : ', y_predict, 'lr :', learning_rate)



# optimizer :  <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x000002291081E250>
# loss :  2.3285017013549805 y_predict :  [[11.289352]] lr : 0.01

# optimizer :  <tensorflow.python.keras.optimizer_v2.adadelta.Adadelta object at 0x0000022910868D90>
# loss :  2.160092830657959 y_predict :  [[10.388709]] lr : 0.01

# optimizer :  <tensorflow.python.keras.optimizer_v2.adagrad.Adagrad object at 0x0000022910870340>
# loss :  2.3745415210723877 y_predict :  [[9.551837]] lr : 0.01

# optimizer :  <tensorflow.python.keras.optimizer_v2.adamax.Adamax object at 0x00000229108702B0>
# loss :  3.210069179534912 y_predict :  [[12.237163]] lr : 0.01

# optimizer :  <tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop object at 0x0000022910870550>
# loss :  1817.342041015625 y_predict :  [[-78.29918]] lr : 0.01

# optimizer :  <tensorflow.python.keras.optimizer_v2.nadam.Nadam object at 0x00000229108703D0>
# loss :  2.580756664276123 y_predict :  [[9.20253]] lr : 0.01