from tabnanny import verbose
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

model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

import time
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
print(start_time)
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)
end_time = time.time() - start_time

print("걸린시간 : ", end_time)

'''
verbose 0
걸린시간 :  11.165835857391357 / 출력없음

verbose 1
걸린시간 :  12.23627257347107 / 출력많음

verbose 2
걸린시간 :  11.604356050491333 / 진행과정바만 없음

verbose 3, 4, 5, 6 ......
걸린시간 :  11.346256494522095 / epoch 만 나옴

'''