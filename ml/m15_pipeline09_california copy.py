import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=666)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

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

# LinearSVR R2 : -6.3395571451731945
# LinearRegression R2 : 0.6161406602616111
# KNeighborsRegressor R2 : 0.1585059607598721
# DecisionTreeRegressor R2 : 0.6221909245352698
# RandomForestRegressor R2 : 0.8239004366598537

# model.score :  0.8046267055831758

