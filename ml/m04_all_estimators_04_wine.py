from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore') # warning 무시
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_wine
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13), (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 0.0
# print(np.max(x_train))  # 1.0

# print(np.min(x_test))  # 1.0
# print(np.max(x_test))  # 1.0


#2. 모델
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

# AdaBoostClassifier 의 정답률 : 0.5370370370370371
# BaggingClassifier 의 정답률 : 0.9814814814814815
# BernoulliNB 의 정답률 : 0.4074074074074074    
# CalibratedClassifierCV 의 정답률 : 0.9444444444444444
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 : 0.7407407407407407   
# DecisionTreeClassifier 의 정답률 : 0.9444444444444444
# DummyClassifier 의 정답률 : 0.4074074074074074ExtraTreeClassifier 의 정답률 : 0.8888888888888888
# ExtraTreesClassifier 의 정답률 : 1.0
# GaussianNB 의 정답률 : 0.9814814814814815
# GaussianProcessClassifier 의 정답률 : 0.37037037037037035
# GradientBoostingClassifier 의 정답률 : 0.9629629629629629
# HistGradientBoostingClassifier 의 정답률 : 1.0KNeighborsClassifier 의 정답률 : 0.6851851851851852
# LabelPropagation 의 정답률 : 0.5185185185185185
# LabelSpreading 의 정답률 : 0.5185185185185185 
# LinearDiscriminantAnalysis 의 정답률 : 0.9814814814814815
# LinearSVC 의 정답률 : 0.9444444444444444
# LogisticRegression 의 정답률 : 0.9629629629629629
# LogisticRegressionCV 의 정답률 : 0.9629629629629629
# MLPClassifier 의 정답률 : 0.05555555555555555
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.8333333333333334  
# NearestCentroid 의 정답률 : 0.6851851851851852NuSVC 의 정답률 : 0.9444444444444444
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.6666666666666666
# Perceptron 의 정답률 : 0.7407407407407407     
# QuadraticDiscriminantAnalysis 의 정답률 : 0.9629629629629629
# RadiusNeighborsClassifier 은 안나온 놈!!!     
# RandomForestClassifier 의 정답률 : 1.0
# RidgeClassifier 의 정답률 : 0.9814814814814815RidgeClassifierCV 의 정답률 : 0.9814814814814815
# SGDClassifier 의 정답률 : 0.6851851851851852  
# SVC 의 정답률 : 0.6666666666666666
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!

#3. 컴파일 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

results= model.score(x_test, y_test)
print('accuracy : ', results)
y_predict = model.predict(x_test)

# LinearSVC의 정확도 : 0.9444444444444444
# LogisticRegression의 정확도 : 0.9629629629629629
# KNeighborsClassifier의 정확도 : 0.6851851851851852
# DecisionTreeClassifier의 정확도 : 0.9629629629629629
# RandomForestClassifier의 정확도 : 1.0