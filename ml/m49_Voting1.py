import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = load_breast_cancer()


x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=123, stratify=datasets.target)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# print(df.head(7))

# 2. 모델
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

model = VotingClassifier(estimators=[('lr', lr), ('knn', knn)], 
                         voting='soft') # hard : 다수결, soft : 확률값

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = accuracy_score(y_test, model.predict(x_test))
print('보팅 결과 : ', round(score, 4))

# 보팅 결과 :  0.9912

classifiers = [lr, knn]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, score2))
    
# LogisticRegression 정확도: 0.9737
# KNeighborsClassifier 정확도: 0.9912