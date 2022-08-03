#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
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
model = LinearSVR()

#3. 컴파일, 훈련
hist = model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , result)
print('r2스코어 : ', r2)

# loss :  -5.753738924806894
# r2스코어 :  -5.753738924806894

# loss :  0.6447481513023376
# r2스코어 :  0.5096276859675669
##################val전후#################  72
# loss :  0.5981025099754333
# r2스코어 :  0.545104755121719   
##################EarlyStopping전후#################
# loss :  0.5845963358879089
# r2스코어 :  0.5505772840253547
##################activation전후#################
# loss :  0.45930835604667664
# r2스코어 :  0.6705258342277447