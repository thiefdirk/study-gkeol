from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델구성
# model = Sequential()
# model.add(Dense(200, input_dim=10))
# model.add(Dense(300))
# model.add(Dense(200))
# model.add(Dense(300,activation='relu'))
# model.add(Dense(150))
# model.add(Dense(180))
# model.add(Dense(1))

input1 = Input(shape=(10,))
dense1 = Dense(200)(input1)
dense2 = Dense(300)(dense1)
dense3 = Dense(200)(dense2)
dense4 = Dense(300, activation='relu')(dense3)
dense5 = Dense(150)(dense4)
dense6 = Dense(180)(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=100,verbose=1,validation_split=0.2, callbacks=[earlyStopping])
print(hist)


end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

# maxabs r2스코어 :  0.6987890148011604    loss :  1988.9619140625

# loss :  2132.549072265625
# r2스코어 :  0.677043978287722