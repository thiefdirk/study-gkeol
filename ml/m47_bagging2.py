from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569,30), (569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66,
                                                    stratify=y)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import BaggingClassifier # 배깅, scaling 해야함
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

model = BaggingClassifier(DecisionTreeClassifier(),
                          n_estimators=200,
                          n_jobs=-1,
                          random_state=123) # baggingclassifier : 분류모델

model.fit(x_train, y_train)


score = model.score(x_test, y_test)
print('score : ', score)


# BaggingClassifier(DecisionTreeClassifier()), standard scaler
# score :  0.9736842105263158

# bagging : 한가지 모델을 여러번 쓰는 것, 
# 기존학습 데이터로부터 랜덤하게 복원추출하여 여러개의 데이터셋을 만들고, 
# 각각의 데이터셋에 대해 모델을 학습시킨다.

#https://sungkee-book.tistory.com/9