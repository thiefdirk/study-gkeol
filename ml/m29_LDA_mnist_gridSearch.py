# n_components > 0.95 이상
# xgboost, gridsearch 또는 randomsearch 사용

# m27_2 결과를 뛰어넘어라

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline # pipeline을 사용하기 위한 함수
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
print(keras.__version__) # 2.9.0
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time

start = time.time() # 시작 시간 체크
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) # (70000, 784)
y= np.append(y_train, y_test) # (70000,)


pca = PCA(n_components=9) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = pca.fit_transform(x) # x를 pca로 변환한다.
# lda = LDA() # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
# x = lda.fit_transform(x, y)
# pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
# cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66, stratify=y) # stratify : 데이터를 군집화할 때 사용하는 열의 값을 지정한다.

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {'n_estimators': [100, 200, 300,], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110,], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree': [0.6, 0.9, 1.0]},
    {'n_estimators': [90, 110,], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree': [0.6, 0.9, 1.0], 'colsample_bylevel': [0.6, 0.9, 1.0]},
]


model = GridSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, n_jobs=1,), parameters,cv=kfold,verbose=True,
                     refit=True,n_jobs=-1,)
# model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, n_jobs=1,), parameters,cv=kfold,verbose=1,)

import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start
#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result) # model.score :  1.0


print("최적의 매개변수 :",model.best_estimator_)


print("최적의 파라미터 :",model.best_params_)

 
print("best_score :",model.best_score_)

print("model_score :",model.score(x_test,y_test))

y_predict = model.predict(x_test)
print('accuracy_score :',accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',accuracy_score(y_test,y_predict))

print("걸린 시간 :",round(end,2),"초")

# model.score :  0.969
# 최적의 매개변수 : XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
#               colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
#               early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, gamma=0, gpu_id=0, grow_policy='depthwise',
#               importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_bin=256, max_cat_to_onehot=4,
#               max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
#               missing=nan, monotone_constraints='()', n_estimators=300,
#               n_jobs=1, num_parallel_tree=1, objective='multi:softprob',
#               predictor='gpu_predictor', random_state=0, reg_alpha=0, ...)
# 최적의 파라미터 : {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 300}
# best_score : 0.9663214285714284
# model_score : 0.969
# accuracy_score : 0.969
# 최적 튠  ACC : 0.969
# 걸린 시간 : 11609.64 초

# PCA 적용 후 결과 // n_components : 10 일때


