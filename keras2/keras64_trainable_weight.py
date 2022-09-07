import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(1))

model.summary()

print(len(model.weights)) # trainable weight
print('====================================')
print(len(model.trainable_weights)) # trainable weight
print('====================================')


model.trainable = False # 훈련을 동결한다. (가중치를 동결한다.)

print(model.weights) # trainable weight
print('====================================')
print(model.trainable_weights) # trainable weight

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)
y_predict = model.predict(x)
print(y_predict[:5])
# 전이학습 : 모델을 저장하고 불러오는 것