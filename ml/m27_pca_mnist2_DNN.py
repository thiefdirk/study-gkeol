# [실습]
# 아까 4가지로 모델을 맹그러봐
# 784개 DNN으로 만들기 (최상의 성능인거 // 0.996이상)과 비교

#time 체크 / fit에서 하고

# 비 지도학습
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
print(keras.__version__) # 2.9.0
import time

start = time.time() # 시작 시간 체크
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) # (70000, 784)
print(x.shape) # (70000, 784)
y= np.append(y_train, y_test) # (70000,)


pca = PCA(n_components=712) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = pca.fit_transform(x) # x를 pca로 변환한다.
pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.
# print('n_components=', 783, ':') # 중요도를 이용해 주요하지 않은 변수를 제거한다.
# print(np.argmax(cumsum >= 0.95)+1) #154
# print(np.argmax(cumsum >= 0.99)+1) #331
# print(np.argmax(cumsum >= 0.999)+1) #486
# print(np.argmax(cumsum+1)) #712
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)


model = RandomForestClassifier()
# model = XGBClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
end = time.time() # 종료 시간 체크

print('실행 시간 :', end-start)
print('accuracy :', result)


###############################################
# [실습]
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95
# 0.99
# 0.999
# 1.0
# 힌트 np.argmax(pca.explained_variance_ratio_)
# pca에 3차원 안들어감 2차원만 들어감
# reshape로 차원을 바꿔준다.
###############################################


#1. 나의 최고의 DNN
# acc스코어 :  0.9746
# 실행시간 :  73.50788712501526

#2. 나의 최고의 CNN
# accuracy :  0.9927374720573425
# 소요 시간 :  0:02:27.802960

#3. pca 0.95
# xgb
# 실행 시간 : 252.82144570350647
# accuracy : 0.9622142857142857

#4. pca 0.99
# rfr
# 실행 시간 : 117.63256239891052
# accuracy : 0.9327142857142857

#5. pca 0.999
# rfr
# 실행 시간 : 150.89516067504883
# accuracy : 0.9142857142857143

#6. pca 1.0
# time=?
# acc=?