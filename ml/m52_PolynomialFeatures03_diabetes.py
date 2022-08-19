from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import time

parameters = [{'classifier__gamma': 0, 'classifier__learning_rate': 0.01,
'classifier__max_depth': None, 'classifier__min_child_weight': 100, 'classifier__n_estimators': 500}]

# n_estimators : [100, 200, 300, 400, 500] / 디폴트 100 / 1~inf / 정수 
# learning_rate : [0.1, 0.2, 0.3, 0.4, 0.5, 1, 0.01, 0.001] / 디폴트 0.3 / 0~1 학습률, 높을수록 더 빠르게 학습됨, 
# max_depth : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 1~inf / 정수 
# gamma : [0, 1, 2, 3, 4, 5, 7, 10, 100] / 디폴트 0 / 0~inf  / 정수 가중치 항등값
# min_child_weight : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] / 디폴트 1 / 0~inf / 정수 
# subsample : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1 / 정수
# colsample_bytree : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1 / 정수
# colsample_bylevel : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1 / 정수
# colsample_bynode : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1 / 정수
# reg_alpha, alpha : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 0 / 0~inf / L1 절대값 가중치 규제
# reg_lambda, lambda : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 1 / 0~inf / L2 절대값 가중치 규제


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape) # (569,)
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=123)

#2. 모델
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

cat = CatBoostRegressor(verbose=0)
lgbm = LGBMRegressor()
xgb = XGBRegressor(gamma=0, learning_rate=0.01, max_depth=None, min_child_weight=100, n_estimators=500)

model = VotingRegressor(estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)]) # hard : 다수결, soft : 확률값, 통상적으로 soft가 더 좋다.


#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train,)
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

score = r2_score(y_test, model.predict(x_test))


classifiers = [cat, lgbm, xgb]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = r2_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, score2))
    
print('보팅 결과 : ', round(score, 4))
print('걸린시간 : ', end)

# CatBoostRegressor 정확도: 0.5382
# LGBMRegressor 정확도: 0.5278
# XGBRegressor 정확도: 0.5561
# 보팅 결과 :  0.5768
# 걸린시간 :  1.1573731899261475

# polynomialfeatures 후
# CatBoostRegressor 정확도: 0.5453
# LGBMRegressor 정확도: 0.5101
# XGBRegressor 정확도: 0.5486
# 보팅 결과 :  0.5583
# 걸린시간 :  4.698806047439575