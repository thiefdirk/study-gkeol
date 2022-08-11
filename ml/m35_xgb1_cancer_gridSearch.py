from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import time

parameters = [
    {'classifier__n_estimators' : [200],
    'classifier__learning_rate' : [0.15],
    'classifier__max_depth' : [5],
    'classifier__gamma' : [0],
    'classifier__min_child_weight' : [0.5],}]

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
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape) # (569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=123, stratify=y)

#2. 모델
from sklearn.pipeline import Pipeline, make_pipeline # pipeline을 사용하기 위한 함수
# pipeline = make_pipeline(MinMaxScaler(), XGBClassifier())
pipeline = Pipeline([('scaler', MinMaxScaler()), ('classifier', XGBClassifier(random_state=123))], verbose=1)

kfold = KFold(n_splits=5, shuffle=True, random_state=123)
model = GridSearchCV(pipeline, parameters, cv=kfold,verbose=1, n_jobs=-1)

#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train,)
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result)


print("최적의 매개변수 :",model.best_estimator_)


print("최적의 파라미터 :",model.best_params_)

 
print("best_score :",model.best_score_)

print("model_score :",model.score(x_test,y_test))

y_predict = model.predict(x_test)
print('accuracy_score :',accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',accuracy_score(y_test,y_predict))

print("걸린 시간 :",round(end,2),"초")

# 최적의 파라미터 : {'classifier__colsample_bytree': 0.5, 'classifier__gamma': 0, 'classifier__learning_rate': 0.15, 
# 'classifier__max_depth': 5, 'classifier__min_child_weight': 0.5, 'classifier__n_estimators': 200, 'classifier__subsample': 1}
# best_score : 0.9670329670329669
# model_score : 1.0
# accuracy_score : 1.0
# 최적 튠  ACC : 1.0
# 걸린 시간 : 2.84 초