# 캐글 자전거 문제풀이
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import datetime as dt

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임        

'''                        
print(train_set)
print(train_set.shape) # (10886, 12)
                  
print(test_set)
print(test_set.shape) # (6493, 9)
print(test_set.info()) # (715, 9)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력
'''


######## 년, 월 ,일 ,시간 분리 ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # 트레인 세트에서 캐주얼 레지스터드 드랍
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) # 테스트 세트에서 데이트타임 드랍

print(train_set)
print(test_set)

##########################################


x = train_set.drop(['count'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.25,
                                                    random_state=31
                                                    )

print(x_train)
print(y_train)

#2. 모델구성

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

no_of_test=[500]
params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}
clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')
clf_rf.fit(x_train,y_train)
pred=clf_rf.predict(x_test)
print((np.sqrt(mean_squared_log_error(pred,y_test))))

clf_rf.best_params_

n_neighbors=[]
for i in range (0,50,5):
    if(i!=0):
        n_neighbors.append(i)
params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
clf_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print((np.sqrt(mean_squared_log_error(pred,y_test))))

'''
params = {
    'n_estimators':(100, 200),
    'max_depth' : (5, 8),
    'min_samples_leaf' : (8, 18),
    'min_samples_split' : (8, 16)
}
model = RandomForestRegressor(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(model, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(x_train, y_train)
 
 
print('최적 하이퍼 파라미터:', grid_cv.best_params_)
print('최적 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
  


model = RandomForestRegressor()
model.fit(x_train, y_train)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


#3. 평가, 예측

y_predict = model.predict(x_test)

def RMSLE(a, b): 
    return np.sqrt(mean_squared_log_error(a, b))

rmsle = RMSLE(y_test, y_predict)
print("RMSLE : ", rmsle)

# RMSLE :  0.3958732766907716

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape) # (6493, 1)

submission_set = pd.read_csv(path + 'sampleSubmission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

print(submission_set)

submission_set['count'] = y_summit
print(submission_set)


submission_set.to_csv(path + 'submission_.csv', index = True)

# https://www.kaggle.com/code/rajmehra03/bike-sharing-demand-rmsle-0-3194/notebook
'''