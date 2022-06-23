from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# print(x)
# print(y)
# print(x.shape, y.shape) #(20640, 8) (20640,)

# print(dataset.feature_names)
# print(dataset.DESCR)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=72
                                                    )

#2. 모델구성

model = Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1500, batch_size=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  0.5870046019554138
# r2스코어 :  0.5535453889641826