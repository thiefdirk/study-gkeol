#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from sklearn.datasets import load_boston
import time

#1. 데이터
datasets = load_boston()
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
from sklearn.svm import LinearSVR, SVC
from sklearn.linear_model import Perceptron, LinearRegression # 로지스틱분류, 분류
from sklearn.neighbors import KNeighborsRegressor # KNN
from sklearn.tree import DecisionTreeRegressor # 의사결정트리
from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트

# model = LinearSVR() 
# model = LinearRegression() # 로지스틱분류
# model = KNeighborsRegressor() 
# model = DecisionTreeRegressor() 
model = RandomForestRegressor() 
#3. 컴파일, 훈련
hist = model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('r2 : ', result)
print('결과 : ', y_predict)


# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', label='loss', color='red')
# plt.plot(hist.history['val_loss'], marker='.', label='val_loss', color='blue')
# plt.grid()
# plt.title('보스턴')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()

# LinearSVR R2 : 0.2576231559485177
# LinearRegression R2 : 0.8111288663608656
# KNeighborsRegressor R2 : 0.5900872726222293
# DecisionTreeRegressor R2 : 0.8025232565231835