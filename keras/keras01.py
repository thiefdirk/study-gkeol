#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(40, input_dim=1))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 https://wikidocs.net/32105
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([4])
print('4의 예측값 : ', result)

# loss :  0.0
# 4의 예측값 :  [[4.]]