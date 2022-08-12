import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer, fetch_california_housing, fetch_covtype, load_wine, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, r2_score


'''
01. iris
02. cancer
# 03. diabetes
04. wine
05. covtype
06. digit
07. kaggle_titanic
'''

parameters = [
    {'classifier__n_estimators' : [100, 200, 300, 400, 500] ,
    'classifier__learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 1, 0.01, 0.001],
    'classifier__max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
    'classifier__min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],}]

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)


# pca = PCA(n_components=6) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
# x = pca.fit_transform(x) # x를 pca로 변환한다.
lda = LDA(n_components=5) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = lda.fit_transform(x, y)
# pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.

# cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(np.unique(y_train, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]) -> (array([0, 1, 2, 3, 4, 5, 6])

# 2. 모델
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, make_pipeline # pipeline을 사용하기 위한 함수
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
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


# fetch_covtype
# model.score :  0.8746848188084645
# model.fit 시간 :  6.83866286277771

# fetch_covtype PCA // n_components=6
# model.score :  0.8365360618916895
# model.fit 시간 :  3.9195711612701416

# fetch_covtype LDA // n_components=6
# model.score :  0.7888694784127777
# model.fit 시간 :  3.6569178104400635

