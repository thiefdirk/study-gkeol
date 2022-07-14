import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import time
start = time.time()

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

x = x.reshape(13, 3, 1)

#2. 모델구성
model = Sequential() 
# cnn은 플래튼까지 써줘서 2차원으로 강제로 맞춰줫는데 이건 그럴필요없음 rnn돌리면 나올때 자동으로 2차원 되서 나옴 
model.add(Bidirectional(GRU(50, return_sequences=True), input_shape=(3, 1)))
model.add(Bidirectional(GRU(85, return_sequences=True)))
model.add(Bidirectional(LSTM(85)))
model.add(Dense(30, activation='relu'))
model.add(Dense(180, activation='swish'))
model.add(Dense(30, activation='relu'))
model.add(Dense(180, activation='swish'))
model.add(Dense(30, activation='relu'))
model.add(Dense(180, activation='swish'))
model.add(Dense(130, activation='relu'))
model.add(Dense(1))

# Bidirectional은 리턴 시퀀스를 사용하지 않기 때문에
# Bidirectional로 씌워줄시 내부에 있는 괄호 안에 리턴 시퀀스 넣어줘야됨


# Bidirectional LSTM
# (features + units)* units * 2 + units * bias(1) * 2
# 결론 : Bidirectional LSTM = 2 * LSTM = simpleRNN * 8

#3. 컴파일
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=400, mode='auto', verbose=1, 
                              restore_best_weights=True)  


model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000, 
                 batch_size=50,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
model.summary()
loss = model.evaluate(x,y)
y_pred = x_predict.reshape(1,3,1)
result = model.predict(y_pred)
print('loss : ', loss)
print('[50,60,70] 의 결과 :', result)
print("time :", time.time() - start)

# loss :  0.0021566241048276424
# [50,60,70] 의 결과 : [[78.02409]]
# time : 31.658336639404297

# 시계열 돌릴때 CONV1D 는 lstm 보다 정확도는 조금 떨어지지만 속도는 빠르기때문에 시계열 돌릴때
# conv1d 로 먼저 뽑아 놓고 lstm이 동일조건에서 더 나은 결과값을 뽑아내는지 확인한다