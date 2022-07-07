from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)

print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델구성

# model = load_model("./_save/keras22_hamsu02_california.h5")

# model = Sequential()
# model.add(Dense(20, input_dim=8,activation='sigmoid'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(8,))
dense1 = Dense(20)(input1)
dense2 = Dense(30, activation='sigmoid')(dense1)
dense3 = Dense(50, activation='relu')(dense2)
dense4 = Dense(50, activation='relu')(dense3)
dense5 = Dense(50, activation='relu')(dense4)
dense6 = Dense(10, activation='relu')(dense5)
output1 = Dense(1)(dense6)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=3000, batch_size=1000,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time()

model.save("./_save/keras22_hamsu02_california.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

# 민맥스 r2스코어 :  0.8129231144815423  loss :  0.26309823989868164

# loss :  0.2707752585411072
# r2스코어 :  0.8057657243030926

# loss :  [0.2925138473510742, 0.33528944849967957]
# r2스코어 :  0.7901720946372207
