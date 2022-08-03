#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
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



end_time = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('acc : ' , results)

# loss :  2452.336669921875
# r2스코어 :  0.6286149246878252
##################val전후#################
# loss :  2211.544189453125
# r2스코어 :  0.6650808519754101
##################EarlyStopping전후#################
# loss :  2170.21484375
# r2스코어 :  0.6713398311092679
##################activation전후#################
# loss :  2162.0830078125
# r2스코어 :  0.6725713563302215