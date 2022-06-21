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
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]])

y = np.transpose(y)
print(y.shape)

#2. 모델구성

model = Sequential()
model.add(Dense(50, input_dim=3))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss )

result = model.predict([[9, 30, 210]])  # 예상 [[10, 1.9, 0]]
print('[9, 30, 210]의 예측값 : ', result)

# loss :  3.3818583489164666e-08
# [9, 30, 210]의 예측값 :  [[ 9.9997244e+00  1.9000002e+00 -1.0118261e-04]]  