import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])

# x = ([x[0:3], x[1:4], x[2:5], x[3:6], x[4:7]])
# print(x)
# y = 



x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
y = np.array([4,5,6,7,8,9,10])

x = x.reshape(7, 3, 1)




print(x.shape) #[7, 3]
print(y.shape) #[7,]

# RNN 에서의 input_shape = (행, 열, 몇개씩 자르는지!!!!!)

x = x.reshape(7, 3, 1)
print(x.shape) #[7, 3, 1]

#2. 모델구성
model = Sequential() 
# cnn은 플래튼까지 써줘서 2차원으로 강제로 맞춰줫는데 이건 그럴필요없음 rnn돌리면 나올때 자동으로 2차원 되서 나옴 
model.add(LSTM(128, input_shape=(3, 1)))
# model.add(SimpleRNN(32))
model.add(Dense(160, activation='relu'))
model.add(Dense(180, activation='swish'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=400, mode='auto', verbose=1, 
                              restore_best_weights=True)  


model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=2000, 
                 batch_size=50,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)
result = model.predict(y_pred)
print('loss : ', loss)
print('[8,9,10] 의 결과 :', result)

# loss :  2.6367613372713095e-06
# [8,9,10] 의 결과 : [[10.8931465]]

#LSTM
# loss :  5.213133044890128e-05
# [8,9,10] 의 결과 : [[10.929763]]

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 128)               66560
# _________________________________________________________________
# dense (Dense)                (None, 160)               20640
# _________________________________________________________________
# dense_1 (Dense)              (None, 180)               28980
# _________________________________________________________________
# dense_2 (Dense)              (None, 180)               32580
# _________________________________________________________________
# dense_3 (Dense)              (None, 140)               25340
# _________________________________________________________________
# dense_4 (Dense)              (None, 140)               19740
# _________________________________________________________________
# dense_5 (Dense)              (None, 140)               19740
# _________________________________________________________________
# dense_6 (Dense)              (None, 40)                5640
# _________________________________________________________________
# dense_7 (Dense)              (None, 1)                 41
# =================================================================
# Total params: 219,261
# Trainable params: 219,261
# Non-trainable params: 0
# _________________________________________________________________