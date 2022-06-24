# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd #pandas : 엑셀땡겨올때 씀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum())
train_set = train_set.dropna()  # train_set 에서 na, null 값 들어간 행 삭제
print(train_set.isnull().sum()) 
print(train_set.shape) # (1328, 10)

############################


x = train_set.drop(['count'], axis=1)  # drop 데이텅서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count'] 
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))
model.add(Dense(110))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1500, batch_size=200, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# y_predict = model.predict(test_set)

# loss :  2847.10302734375
# RMSE :  51.6441636904107

# 함수에 대해서 정의, 공부
# https://wikidocs.net/63
# https://ordo.tistory.com/59