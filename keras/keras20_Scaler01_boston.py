# 과제 : MaxAbsScaler, RobustScaler 정리하기, 12개 파일에 적용해보기
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import time

datasets = load_boston()
x = datasets.data
y = datasets['target']

# print(np.min(x))  # 0.0
# print(np.max(x))  # 711.0
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# print(x[:10])

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.7, random_state=66
)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

model.fit(x_train, y_train, epochs=3000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측


loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)


# 스케일러 하기전 r2스코어 :  0.8098177492286052  loss :  15.895992279052734
# 민맥스 r2스코어 :  0.8816872764076906  loss :  10.058992385864258
# 스탠다드 r2스코어 :  0.8469140436844695    loss :  12.6475191116333

# maxabs r2스코어 :  0.8688676206779293    loss :  10.83377742767334
# robust r2스코어 :  0.8804021674574816    loss :  9.880825996398926