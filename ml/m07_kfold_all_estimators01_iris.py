import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩

from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='classifier')

#('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

#print('allAlgorithms : ', allAlgorithms)
#print('모델갯수 : ', len(allAlgorithms)) # 모델갯수 :  41

for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm()
        # model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('Model Name : ', name)
        #print('ACC : ', scores) 
        print('cross_val_score : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        print(y_predict)
        
        #y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')
    
    
    
'''
Model Name :  AdaBoostClassifier
cross_val_score :  0.9083
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  BaggingClassifier
cross_val_score :  0.9417
[1 2 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  BernoulliNB
cross_val_score :  0.2833
[2 2 2 2 2 2 1 2 0 2 2 1 2 1 2 1 2 2 2 1 1 2 2 1 2 2 2 2 1 1]
Model Name :  CalibratedClassifierCV
cross_val_score :  0.8667
[1 1 2 0 2 2 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 2 2 2 2 0 0 1 2]
Model Name :  CategoricalNB
cross_val_score :  nan
CategoricalNB 은 실행되지 않는다.
ClassifierChain 은 실행되지 않는다.
Model Name :  ComplementNB
cross_val_score :  0.6667
[2 2 2 0 2 2 0 0 0 2 2 2 0 2 2 0 2 2 2 2 0 2 2 2 2 2 0 2 2 2]
Model Name :  DecisionTreeClassifier
cross_val_score :  0.925
[1 1 1 0 1 1 0 0 0 2 2 2 0 1 2 0 1 2 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  DummyClassifier
cross_val_score :  0.2583
[2 2 2 2 2 2 1 2 2 2 2 1 2 1 2 1 2 2 2 1 1 2 2 1 2 2 2 2 1 1]
Model Name :  ExtraTreeClassifier
cross_val_score :  0.9083
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 1 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  ExtraTreesClassifier
cross_val_score :  0.9417
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  GaussianNB
cross_val_score :  0.95
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  GaussianProcessClassifier
cross_val_score :  0.9
[1 2 2 0 2 1 0 0 0 2 2 2 0 1 2 0 1 2 2 2 0 1 2 1 2 2 0 0 1 2]
Model Name :  GradientBoostingClassifier
cross_val_score :  0.9333
[1 2 1 0 1 2 0 0 0 2 2 2 0 1 2 0 1 2 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  HistGradientBoostingClassifier
cross_val_score :  0.9417
[2 2 2 2 2 2 1 2 2 2 2 1 2 1 2 1 2 2 2 1 1 2 2 1 2 2 2 2 1 1]
Model Name :  KNeighborsClassifier
cross_val_score :  0.9667
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  LabelPropagation
cross_val_score :  0.9583
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  LabelSpreading
cross_val_score :  0.9583
[1 1 1 0 1 1 0 0 0 2 2 2 0 1 2 0 1 1 2 2 0 1 1 1 1 2 0 0 1 2]
Model Name :  LinearDiscriminantAnalysis
cross_val_score :  0.975
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 2 2 0 0 2 2]
Model Name :  LinearSVC
cross_val_score :  0.9167
[1 1 2 0 2 2 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 2 2 2 2 0 0 1 2]
Model Name :  LogisticRegression
cross_val_score :  0.9083
[1 2 2 0 2 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 2 2 2 2 0 0 1 2]
Model Name :  LogisticRegressionCV
cross_val_score :  0.9417
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  MLPClassifier
cross_val_score :  0.8833
[1 2 2 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 2 2 1 2 0 0 1 2]
MultiOutputClassifier 은 실행되지 않는다.
Model Name :  MultinomialNB
cross_val_score :  0.5917
[2 2 2 0 2 2 0 0 0 2 2 1 0 1 2 0 2 2 2 1 0 2 2 1 2 2 0 2 1 1]
Model Name :  NearestCentroid
cross_val_score :  0.9083
[1 1 1 0 1 1 0 0 0 2 2 2 0 1 1 0 1 1 2 2 0 1 1 1 1 2 0 0 1 2]
Model Name :  NuSVC
cross_val_score :  0.95
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
OneVsOneClassifier 은 실행되지 않는다.
OneVsRestClassifier 은 실행되지 않는다.
OutputCodeClassifier 은 실행되지 않는다.
Model Name :  PassiveAggressiveClassifier
cross_val_score :  0.9167
[1 2 2 0 2 2 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 0 2 2 1 2 0 0 1 2]
Model Name :  Perceptron
cross_val_score :  0.7583
[0 2 2 0 2 0 0 0 0 2 2 2 0 1 2 0 0 2 2 1 0 2 2 2 1 2 0 1 2 2]
Model Name :  QuadraticDiscriminantAnalysis
cross_val_score :  0.9667
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 2 1 2 2 2 0 1 2 2]
Model Name :  RadiusNeighborsClassifier
cross_val_score :  0.4083
[2 2 2 1 2 2 0 0 0 2 2 1 1 1 2 0 2 2 2 1 0 2 2 1 2 2 0 0 1 1]
Model Name :  RandomForestClassifier
cross_val_score :  0.95
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
Model Name :  RidgeClassifier
cross_val_score :  0.8333
[2 2 2 0 2 2 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 2 2 2 2 0 0 1 2]
Model Name :  RidgeClassifierCV
cross_val_score :  0.825
[2 2 2 0 2 2 0 0 0 2 2 2 0 2 2 0 1 2 2 2 0 1 2 2 2 2 0 0 1 2]
Model Name :  SGDClassifier
cross_val_score :  0.85
[1 1 0 0 2 2 0 0 0 2 2 2 0 2 2 0 0 2 2 2 0 1 0 2 2 2 0 1 1 2]
Model Name :  SVC
cross_val_score :  0.95
[1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
StackingClassifier 은 실행되지 않는다.
VotingClassifier 은 실행되지 않는다.
'''