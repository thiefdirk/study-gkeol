from sklearn.datasets import load_diabetes, load_wine, fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor


datasets = fetch_covtype()
x = datasets.data
y = datasets.target

le = LabelEncoder()
y = le.fit_transform(y)

print(x.shape, y.shape) # (569,30), (569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor # 배깅, scaling 해야함
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

# model = XGBClassifier()


model = BaggingClassifier(XGBClassifier(),
                          n_estimators=200,
                          n_jobs=-1,
                          random_state=123) # baggingclassifier : 분류모델

model.fit(x_train, y_train)


score = model.score(x_test, y_test)
print('score : ', score)

#########################bagging 안한거#############################
# score :  0.869392356479609


#########################bagging 한거#############################
# score :  1.0