import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)])
# print(range(10)) # range(10) 0 부터 10 미만
# for i in range(10):
#     print(i)

print(x.shape) # (3, 10)

x = np.transpose(x)

print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])

y = np.transpose(y)
print(y.shape)

#2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(35))
model.add(Dense(4))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss )

result = model.predict([[9, 30, 210]])  # 예상 [[10, 1.9]]
print('[9, 30, 210]의 예측값 : ', result)

# loss :  1.9741741574819116e-09
# [9, 30, 210]의 예측값 :  [[9.999998  1.9000663]]