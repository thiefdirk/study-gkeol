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
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시


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
allAlgorithms = all_estimators(type_filter='classifier') # 분류모델만 추출
# allAlgorithms = all_estimators(type_filter='regresser') # 회귀모델만 추출
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

print(allAlgorithms) # 모든 모델을 보여줌
print(len(allAlgorithms)) # 모든 모델의 갯수를 보여줌, 총 갯수는 총 모델의 갯수 + 1, 41


for (name, algorithm) in allAlgorithms: # key, value로 나누어서 보여줌
    try: 
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 :', accuracy_score(y_test, y_predict))
    except: # 모델에 오류가 있을 경우 예외를 발생시킴
        print(name, '은 안나온 놈!!!')
        continue    # 예외가 발생하면 다음 모델으로 넘어가게 하는 코드
# TypeError: __init__() missing 1 required positional argument: 'base_estimator', 이런 에러가 뜸
# 예외처리 해야함

#3. 컴파일, 훈련
hist = model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , result)
print('r2스코어 : ', r2)

# LinearSVR R2 : -6.3395571451731945
# LinearRegression R2 : 0.6161406602616111
# KNeighborsRegressor R2 : 0.1585059607598721
# DecisionTreeRegressor R2 : 0.6221909245352698
# RandomForestRegressor R2 : 0.8239004366598537