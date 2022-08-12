from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import time
from catboost import CatBoostClassifier, CatBoostRegressor

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
# model = GridSearchCV(pipeline, parameters, cv=kfold,verbose=1, n_jobs=-1)
model = XGBClassifier(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)
    
#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train, early_stopping_rounds=10, 
          eval_set=[(x_train,y_train), (x_test,y_test)], eval_metric='error', verbose=1)
        #   eval_set=[(x_test,y_test)])
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result)

y_predict = model.predict(x_test)

print('accuracy_score :',accuracy_score(y_test,y_predict))

print('========================================================')
hist = model.evals_result()
print(hist)

import matplotlib.pyplot as plt

# plt.plot(hist['validation_0']['error'])
# plt.plot(hist['validation_1']['error'])
# plt.xlabel('round')# x축 이름 설정
# plt.ylabel('error')# y축 이름 설정
# plt.title('XGBoost')# 차트 제목 설정
# plt.legend(['train', 'test']) # 범례 설정
# plt.show()


print(hist)
print(hist['validation_0'])


#subplot
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.xlabel('round')# x축 이름 설정
    plt.ylabel('error')# y축 이름 설정
    if i == 0:
        plt.plot(hist['validation_'+str(i)]['error'],'b')
        plt.legend(['train'])
        plt.title('XGBoost')# 차트 제목 설정
    else:
        plt.plot(hist['validation_'+str(i)]['error'],'r')
        plt.legend(['test'])
    # 색상 설정
    plt.grid(True)
    
plt.show()
