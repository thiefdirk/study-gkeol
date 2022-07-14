import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, Dropout, GRU
from sklearn.model_selection import train_test_split

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
size1 = 5
size2 = 4
print(x_predict.shape)
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size1)
print(bbb)
print(bbb.shape) # (6, 5)

x = bbb[:,:-1]
y = bbb[:,-1]
print(x,y)
print(x.shape, y.shape) # (96, 4) (96,)

x = x.reshape(96, 4, 1)

ccc = split_x(x_predict, size2)
print(ccc)
print(ccc.shape) # (7, 4)

# 모델구성 및 평가 예측할것


#2. 모델구성
model = Sequential() 
# cnn은 플래튼까지 써줘서 2차원으로 강제로 맞춰줫는데 이건 그럴필요없음 rnn돌리면 나올때 자동으로 2차원 되서 나옴 
# model.add(SimpleRNN(units=100, input_shape=(3, 1)))  #(units, input_shape=(batch, timesteps, feature))
model.add(GRU(200, return_sequences=True, input_shape=(4,1))) #return_seqeunces=True (N,3,1) -> (N,3,10)
model.add(GRU(250, return_sequences=True))
model.add(GRU(250))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='swish'))
model.add(Dense(150, activation='swish'))
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
                 batch_size=100,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측

loss = model.evaluate(x,y)
y_pred = ccc.reshape(7,4,1)
result = model.predict(y_pred)
print('loss : ', loss)
print('[96~104] 의 결과 :', result)

# loss :  6.098812445998192e-05
# [96~104] 의 결과 :
# [[ 99.95826 ]
#  [100.810196]
#  [101.592155]
#  [102.32227 ]
#  [103.001656]
#  [103.6343  ]
#  [104.22374 ]]