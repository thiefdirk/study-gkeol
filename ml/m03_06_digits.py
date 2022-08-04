import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_digits
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR
print(sk.__version__)

#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(x,y)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# print(y_test)
# print(y_train)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 로지스틱분류, 분류
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier # 의사결정트리
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트

# model = LinearSVC() 
# model = LogisticRegression() # 로지스틱분류
# model = KNeighborsClassifier() 
# model = DecisionTreeClassifier() 
model = RandomForestClassifier() 

#3. 컴파일 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

results= model.score(x_test, y_test)
print('accuracy : ', results)


y_predict = model.predict(x_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

# LinearSVC : 0.9555555555555556
# LogisticRegression : 0.9703703703703703
# KNeighborsClassifier : 0.9944444444444445
# DecisionTreeClassifier : 0.85
# RandomForestClassifier : 0.9555555555555556