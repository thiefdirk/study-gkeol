from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import time

#1. 데이터
datasets = load_breast_cancer()
# print(datasets) (569,30)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data   #['data']
y = datasets.target #['target']
print(x.shape, y.shape) # (569,30), (569,)


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

# AdaBoostClassifier 의 정답률 : 0.9532163742690059
# BaggingClassifier 의 정답률 : 0.9649122807017544
# BernoulliNB 의 정답률 : 0.6432748538011696
# CalibratedClassifierCV 의 정답률 : 0.8947368421052632
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 : 0.8888888888888888   
# DecisionTreeClassifier 의 정답률 : 0.9415204678362573
# DummyClassifier 의 정답률 : 0.6432748538011696ExtraTreeClassifier 의 정답률 : 0.935672514619883
# ExtraTreesClassifier 의 정답률 : 0.9707602339181286
# GaussianNB 의 정답률 : 0.9473684210526315
# GaussianProcessClassifier 의 정답률 : 0.8947368421052632
# GradientBoostingClassifier 의 정답률 : 0.9590643274853801
# HistGradientBoostingClassifier 의 정답률 : 0.9707602339181286
# KNeighborsClassifier 의 정답률 : 0.9064327485380117
# LabelPropagation 의 정답률 : 0.3684210526315789
# LabelSpreading 의 정답률 : 0.3684210526315789
# LinearDiscriminantAnalysis 의 정답률 : 0.9649122807017544
# LinearSVC 의 정답률 : 0.8070175438596491
# LogisticRegression 의 정답률 : 0.9239766081871345
# LogisticRegressionCV 의 정답률 : 0.935672514619883
# MLPClassifier 의 정답률 : 0.9181286549707602
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.8830409356725146  
# NearestCentroid 의 정답률 : 0.8713450292397661NuSVC 의 정답률 : 0.8713450292397661
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.8654970760233918
# Perceptron 의 정답률 : 0.8304093567251462     
# QuadraticDiscriminantAnalysis 의 정답률 : 0.9473684210526315
# RadiusNeighborsClassifier 은 안나온 놈!!!     
# RandomForestClassifier 의 정답률 : 0.9707602339181286
# RidgeClassifier 의 정답률 : 0.9649122807017544RidgeClassifierCV 의 정답률 : 0.9649122807017544
# SGDClassifier 의 정답률 : 0.8128654970760234  
# SVC 의 정답률 : 0.8888888888888888
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
            
#3. 컴파일, 훈련

hist = model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('acc : ', result)
print('결과 : ', y_predict)

# LinearSVC : 0.9766081871345029
# LogisticRegression : 0.9883040935672515
# KNeighborsClassifier : 0.9590643274853801
# DecisionTreeClassifier : 0.9590643274853801
# RandomForestClassifier : 0.9649122807017544