import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


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
cat = CatBoostClassifier(verbose=0)
lgbm = LGBMClassifier()
xgb = XGBClassifier()

model = VotingClassifier(estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)],
                         voting='soft') # hard : 다수결, soft : 확률값, 통상적으로 soft가 더 좋다.

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
score = accuracy_score(y_test, model.predict(x_test))

# 보팅 결과 :  0.9912

classifiers = [cat, lgbm, xgb]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, score2))
    
print('보팅 결과 : ', round(score, 4))

# CatBoostClassifier 정확도: 0.9912
# LGBMClassifier 정확도: 0.9825
# XGBClassifier 정확도: 0.9912
# 보팅 결과 :  0.9825