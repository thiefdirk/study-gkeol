from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=100, verbose = 1, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# loss :  0.6447481513023376
# r2스코어 :  0.5096276859675669
##################val전후#################
# loss :  0.5981025099754333
# r2스코어 :  0.545104755121719