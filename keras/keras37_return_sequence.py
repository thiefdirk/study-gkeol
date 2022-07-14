# https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, Dropout, GRU
from sklearn.model_selection import train_test_split

#1. 데이터
# x = ([x[0:3], x[1:4], x[2:5], x[3:6], x[4:7]])
# print(x)
# y = 

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape) #[7, 3]
print(y.shape) #[7,]

x = x.reshape(13, 3, 1)

print(x.shape) #[7, 3]
print(y.shape) #[7,]

# RNN 에서의 input_shape = (행, 열, 몇개씩 자르는지!!!!!)


print(x.shape) #[7, 3, 1]

#2. 모델구성
model = Sequential() 
# cnn은 플래튼까지 써줘서 2차원으로 강제로 맞춰줫는데 이건 그럴필요없음 rnn돌리면 나올때 자동으로 2차원 되서 나옴 
# model.add(SimpleRNN(units=100, input_shape=(3, 1)))  #(units, input_shape=(batch, timesteps, feature))
model.add(LSTM(100, return_sequences=True, input_shape=(3,1))) #return_seqeunces=True (N,3,1) -> (N,3,10)
model.add(LSTM(150))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='swish'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='swish'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='swish'))
model.add(Dense(1))

model.summary()

# 결론 : LSTM = simpleRNN * 4
# 숫자 4의 의미는 cell state, input gate, output gate, forget gate
# (features + units)* units + units * bias(1)








#3. 컴파일
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=400, mode='auto', verbose=1, 
                              restore_best_weights=True)  


model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1500, 
                 batch_size=50,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

loss = model.evaluate(x,y)
y_pred = x_predict.reshape(1,3,1)
result = model.predict(y_pred)
print('loss : ', loss)
print('[50,60,70] 의 결과 :', result)

# loss :  0.00028452574042603374
# [50,60,70] 의 결과 : [[78.450356]]