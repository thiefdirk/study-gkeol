import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 https://kau-deeperent.tistory.com/130 스칼라, 벡터, 행렬, 텐서 설명
# x = np.array([[1,2,3,4,5,6,7,8,9,10],
#              [1, 1, 1, 1, 2, 1.3 ,1.4 ,1.5 ,1.6 ,1.4]])  (2,10) 행, 열 바뀜
# y = np.array([11,12,13,14,15,16,17,18,19,20]) (10, ) 
# print(np.shape(x)) x라는 행렬모양 출력

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3 ,1.4 ,1.5 ,1.6 ,1.4],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([11,12,13,14,15,16,17,18,19,20]) # (10, )

# x = x.T  행과 열 통째로 변경
# x = x.transpose() 상동함
# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

x = x.T

print(x)
print(x.shape)   #(10, 2)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(35))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss )

result = model.predict([[10, 1.4, 0]])  # 괄호 하나빼먹으면 (2, )나오기때문에 에러
print('[10, 1.4, 0]의 예측값 : ', result)

# loss :  9.267793211620301e-05
# [10, 1.4, 0]의 예측값 :  [[19.990023]]


# [숙제] 모델을 완성하시오 예측 : [[10, 1.4, 0]]
# 과제 리스트 내용정리, 예제 5개