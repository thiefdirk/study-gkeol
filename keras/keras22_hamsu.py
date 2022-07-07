import numpy as np


#데이터 https://kau-deeperent.tistory.com/130 스칼라, 벡터, 행렬, 텐서 설명
# x = np.array([[1,2,3,4,5,6,7,8,9,10],
#              [1, 1, 1, 1, 2, 1.3 ,1.4 ,1.5 ,1.6 ,1.4]])  (2,10) 행, 열 바뀜
# y = np.array([11,12,13,14,15,16,17,18,19,20]) (10, ) 
# print(np.shape(x)) x라는 행렬모양 출력


#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3 ,1.4 ,1.5 ,1.6 ,1.4],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([11,12,13,14,15,16,17,18,19,20]) # (10, )

# x = x.T  행과 열 통째로 변경
# x = x.transpose() 상동함
# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈
print(x.shape)   #(10, 3)
print(y.shape)

x = x.T

print(x.shape)   #(3, 10)
print(y.shape)


#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(10, input_dim=3))
# model.add(Dense(10, input_shape=(3,))) # input_dim = 3 은 input_shape=(3,)로도 명시가능
# model.add(Dense(5, activation='relu'))
# model.add(Dense(3, activation='sigmoid'))
# model.add(Dense(1))

input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation='relu')(dense1)
dense3 = Dense(3, activation='sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=10, batch_size=1)
