# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
train_set = train_set.fillna(train_set.mean()) # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값
print(train_set.isnull().sum()) 
print(train_set.shape) # (1328, 10)

############################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count'] 
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )

#2. 모델구성

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline # pipeline을 사용하기 위한 함수

# model = SVC()
model = make_pipeline(MinMaxScaler(), RandomForestRegressor()) # pipeline을 사용하면 여러개의 모델을 한번에 학습시키기 때문에 성능이 좋아진다.

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result) 

# r2스코어 :  0.6710084389225455

# model.score :  0.7836562663507497
