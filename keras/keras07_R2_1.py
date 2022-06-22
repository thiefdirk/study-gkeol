import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,8,9,10,8,12,13,11,14,15,16,18,17,20])

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(51))
model.add(Dense(50))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)













# import matplotlib.pyplot as plt
# 그림그릴때 불러오는 기능
# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()