import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)

print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''
#2. 모델구성
# model = Sequential()
# model.add(Dense(20, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1))

# model = Sequential()
# model.add(Dense(20, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1))

model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=750, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  3.0837433338165283
# r2스코어 :  0.7566761315811648
# 700

# loss :  19.6019344329834
# r2스코어 :  0.7627375911687064
# 750