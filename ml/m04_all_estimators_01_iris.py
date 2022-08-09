import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore') # warning 무시

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



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

# AdaBoostClassifier 의 정답률 : 0.6333333333333333
# BaggingClassifier 의 정답률 : 0.9666666666666667
# BernoulliNB 의 정답률 : 0.4
# CalibratedClassifierCV 의 정답률 : 0.9666666666666667
# CategoricalNB 의 정답률 : 0.3333333333333333  
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 : 0.6666666666666666   
# DecisionTreeClassifier 의 정답률 : 0.9666666666666667
# DummyClassifier 의 정답률 : 0.3
# ExtraTreeClassifier 의 정답률 : 0.8666666666666667
# ExtraTreesClassifier 의 정답률 : 0.9333333333333333
# GaussianNB 의 정답률 : 0.9666666666666667
# GaussianProcessClassifier 의 정답률 : 0.9666666666666667
# GradientBoostingClassifier 의 정답률 : 0.9666666666666667
# HistGradientBoostingClassifier 의 정답률 : 0.8666666666666667
# KNeighborsClassifier 의 정답률 : 1.0
# LabelPropagation 의 정답률 : 0.9666666666666667
# LabelSpreading 의 정답률 : 0.9666666666666667 
# LinearDiscriminantAnalysis 의 정답률 : 1.0    
# LinearSVC 의 정답률 : 0.9666666666666667      
# LogisticRegression 의 정답률 : 0.9666666666666667
# LogisticRegressionCV 의 정답률 : 1.0
# MLPClassifier 의 정답률 : 0.9333333333333333
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.6333333333333333  
# NearestCentroid 의 정답률 : 0.9666666666666667NuSVC 의 정답률 : 0.9666666666666667
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.9333333333333333
# Perceptron 의 정답률 : 0.9333333333333333     
# QuadraticDiscriminantAnalysis 의 정답률 : 1.0 
# RadiusNeighborsClassifier 의 정답률 : 0.4666666666666667
# RandomForestClassifier 의 정답률 : 0.9666666666666667
# RidgeClassifier 의 정답률 : 0.9333333333333333RidgeClassifierCV 의 정답률 : 0.8333333333333334
# SGDClassifier 의 정답률 : 0.8333333333333334  
# SVC 의 정답률 : 1.0
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!

#3. 컴파일 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

y_predict = model.predict(x_test)
acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', label='loss', color='red')
# plt.plot(hist.history['val_loss'], marker='.', label='val_loss', color='blue')
# plt.grid()
# plt.title('시그모이드')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()

# loss :  0.05714249983429909
# accuracy :  1.0

# LinearSVC 모델의 정확도 :  0.9666666666666667
# LogisticRegression 모델의 정확도 :  1.0
# KNN 모델의 정확도 :  0.9666666666666667
# DecisionTreeClassifier 모델의 정확도 :  0.9333333333333333
# RandomForestClassifier 모델의 정확도 :  0.9333333333333333