from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
import numpy as np
import time

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

start_time = time.time()

# #2. 모델구성
# model = Sequential()
# model.add(Dense(256, input_dim=13))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
# model.summary()


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
#                               restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping, mcp],
#                  verbose=1)

end_time = time.time() - start_time

model = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)

# loss :  9.568774223327637
# r2스코어 :  0.8855176121925235
# 걸린시간 :  20.815696001052856

# loss :  9.568774223327637
# r2스코어 :  0.8855176121925235
# 걸린시간 :  0.0