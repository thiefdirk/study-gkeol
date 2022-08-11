import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer # 이터러블 입력시 사용하는 모듈 추가
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 



parameters = [
    {'XGB__n_estimators' : [100, 200], 'XGB__max_depth': [40,30,20,50], 'XGB__min_samples_leaf':[15, 30, 50, 100]}, #epochs
    {'XGB__max_depth' : [6, 8, 10, 12], 'XGB__min_samples_split':[2, 4, 5, 20], 'XGB__n_jobs' : [-1, 3, 5]},
    {'XGB__min_samples_leaf' : [3, 5, 7, 10], 'XGB__n_estimators':[150, 300, 200], 'XGB__max_depth':[7, 8, 9, 10]},
    {'XGB__min_samples_split' : [2, 3, 5, 10]},
    {'XGB__n_jobs' : [-1, 2, 4]}] 
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 knn 임퓨터 ####
# imputer1 = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3
# imputer2 = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3

imputer1 = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001)
imputer2 = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001)

print(train_set.isnull().sum())
imputer1.fit(train_set) # 데이터프레임에 적용하기 위해 fit()함수 사용
imputer2.fit(test_set) # 데이터프레임에 적용하기 위해 fit()함수 사용
train_set_imputer = imputer1.transform(train_set) # transform()함수를 사용하여 데이터프레임을 적용하기 위해 transform()함수 사용
test_set_imputer = imputer2.transform(test_set) # transform()함수를 사용하여 데이터프레임을 적용하기 위해 transform()함수 사용
print(train_set.shape) # (1459, 10)
train_set = pd.DataFrame(train_set_imputer, columns=train_set.columns) # 데이터프레임을 데이터프레임으로 변환하기 위해 DataFrame()함수 사용
test_set = pd.DataFrame(test_set_imputer, columns=test_set.columns) # 데이터프레임을 데이터프레임으로 변환하기 위해 DataFrame()함수 사용
############################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count'] 
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )


#2. 모델구성

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor # xgboost 사용
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline, Pipeline # pipeline을 사용하기 위한 함수

pipe = Pipeline([('minmax', MinMaxScaler()), ('XGB', XGBRegressor())], verbose=1)
# pipe = make_pipeline(MinMaxScaler(), XGBRegressor())
model = GridSearchCV(pipe, parameters,verbose=1,cv=kfold,
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
print('accuracy_score :',r2_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',r2_score(y_test,y_predict))

print("걸린 시간 :",round(end,2),"초")

# 최적의 파라미터 : {'XGB__max_depth': 6, 'XGB__min_samples_split': 2, 'XGB__n_jobs': -1}     
# best_score : 0.7575992901520798
# model_score : 0.8014237720454513
# accuracy_score : 0.8014237720454513
# 최적 튠  ACC : 0.8014237720454513
# 걸린 시간 : 45.42 초