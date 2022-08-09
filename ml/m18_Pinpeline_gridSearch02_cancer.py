import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=666)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {'RF__n_estimators' : [100, 200], 'RF__max_depth': [40,30,20,50], 'RF__min_samples_leaf':[15, 30, 50, 100]}, #epochs
    {'RF__max_depth' : [6, 8, 10, 12], 'RF__min_samples_split':[2, 4, 5, 20], 'RF__n_jobs' : [-1, 3, 5]},
    {'RF__min_samples_leaf' : [3, 5, 7, 10], 'RF__n_estimators':[150, 300, 200], 'RF__max_depth':[7, 8, 9, 10]},
    {'RF__min_samples_split' : [2, 3, 5, 10]},
    {'RF__n_jobs' : [-1, 2, 4]}] 

# 각 횟수를 병렬로 진행해 총 42번을  1회에 한다.
#rbf= Gaussian basis function RBF 뉴럴네트워크의 경우 각 데이터에 맞는 
# Kernel function을 이용하기에 비선형적이고, MLP보다 학습이 빠르다.

#2. 모델구성

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline, Pipeline # pipeline을 사용하기 위한 함수

pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())], verbose=1)

# model = SVC()
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) # pipeline을 사용하면 여러개의 모델을 한번에 학습시키기 때문에 성능이 좋아진다.
model = GridSearchCV(pipe, parameters,cv=kfold,verbose=1,
                     refit=True,n_jobs=-1,)


#3. 컴파일,훈련
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

# model.score :  0.9210526315789473
# 최적의 매개변수 : Pipeline(steps=[('minmax', MinMaxScaler()),
#                 ('RF',
#                  RandomForestClassifier(max_depth=6, min_samples_split=4,
#                                         n_jobs=5))],
#          verbose=1)
# 최적의 파라미터 : {'RF__max_depth': 6, 'RF__min_samples_split': 4, 'RF__n_jobs': 5}
# best_score : 0.9692307692307693
# model_score : 0.9210526315789473
# accuracy_score : 0.9210526315789473
# 최적 튠  ACC : 0.9210526315789473
# 걸린 시간 : 18.77 초