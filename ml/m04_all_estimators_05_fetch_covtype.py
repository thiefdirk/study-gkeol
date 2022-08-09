from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import fetch_covtype
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]

#cross_validation 

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
# allAlgorithms = all_estimators(type_filter='regressor') # 회귀모델만 추출
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

# AdaBoostClassifier 의 정답률 : 0.5342160822471085
# BaggingClassifier 의 정답률 : 0.9588248118230218
# BernoulliNB 의 정답률 : 0.6303412428859922    
# BaggingClassifier 의 정답률 : 0.9588248118230218
# BernoulliNB 의 정답률 : 0.6303412428859922    
# CalibratedClassifierCV 의 정답률 : 0.6684470809619975
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 은 안나온 놈!!!
# DecisionTreeClassifier 의 정답률 : 0.9349297778593721
# DummyClassifier 의 정답률 : 0.48640880301083167
# ExtraTreeClassifier 의 정답률 : 0.8503419313383513
# ExtraTreesClassifier 의 정답률 : 0.9509305581053791
# GaussianNB 의 정답률 : 0.46058610244171105    
# GaussianProcessClassifier 은 안나온 놈!!!     
# GradientBoostingClassifier 의 정답률 : 0.7742679456581605
# HistGradientBoostingClassifier 의 정답률 : 0.7824834771433816
# KNeighborsClassifier 의 정답률 : 0.9659847163576281
# LabelPropagation 은 안나온 놈!!!
# LabelSpreading 은 안나온 놈!!!
# LinearDiscriminantAnalysis 의 정답률 : 0.6787451808334863
# LinearSVC 의 정답률 : 0.32425532403157703     
# LogisticRegression 의 정답률 : 0.6187695061501745
# LogisticRegressionCV 의 정답률 : 0.6709599320727005
# MLPClassifier 의 정답률 : 0.77542684046264    
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 은 안나온 놈!!!
# NearestCentroid 의 정답률 : 0.19382802460069762
# NuSVC 은 안나온 놈!!!
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.41645630622360935
# Perceptron 의 정답률 : 0.5785409399669543     
# QuadraticDiscriminantAnalysis 의 정답률 : 0.08309046263998532
# RadiusNeighborsClassifier 은 안나온 놈!!!     
# RandomForestClassifier 의 정답률 : 0.9525082614283091
# RidgeClassifier 의 정답률 : 0.7003166880851845RidgeClassifierCV 의 정답률 : 0.7005117495869286
# SGDClassifier 의 정답률 : 0.5229943087938315  
# SVC 의 정답률 : 0.7124391867082798
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!

#3. 컴파일 훈련

model.fit(x_train, y_train)

#4. 평가, 예측

results= model.score(x_test, y_test)
print('accuracy : ', results)


y_predict = model.predict(x_test)


acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

# LinearSVC accuracy : 0.421826234624564
# LogisticRegression accuracy : 0.6187695061501745
# KNeighborsClassifier accuracy : 0.9659847163576281
# DecisionTreeClassifier accuracy : 0.935377271892785
# RandomForestClassifier accuracy : 0.952967229667707
