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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



parameters_xgb = [
    {'classifier__n_estimators' : [100, 200, 300, 400, 500] ,
    'classifier__learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 1, 0.01, 0.001],
    'classifier__max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
    'classifier__min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],}]

# parameters_xgb = [
#     {'gamma': 4, 'learning_rate': 0.1, 'max_depth': 2, 'min_child_weight': 0, 
#     'n_estimators': 100}]

parameters_rfr = [{
    'RFR__bootstrap': [True], 'RFR__max_depth': [5, 10, None], 
    'RFR__max_features': ['auto', 'log2'], 'RFR__n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]}]

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

#1. 데이터
path = './_data/dacon_travel/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (1955, 19)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape)  # (2933, 18)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(test_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

x = train_set.iloc[:, :-1] # 컬럼을 제외한 모든 컬럼을 x로 저장
y = train_set.iloc[:, -1] # 마지막 컬럼을 y로 저장

# train set 과 test set concat하기
all_data = pd.concat([x, test_set], axis=0) # axis=0은 행을 더하는 것이라는 뜻

le_TypeofContact = LabelEncoder() # TypeofContact 컬럼을 인코딩해줌
le_Occupation = LabelEncoder() # Occupation 컬럼을 인코딩해줌
le_gender = LabelEncoder() # 성별 컬럼을 인코딩해줌
le_ProductPitched = LabelEncoder() # ProductPitched 컬럼을 인코딩해줌
le_MaritalStatus = LabelEncoder() # MaritalStatus 컬럼을 인코딩해줌
le_Designation = LabelEncoder() # Designation 컬럼을 인코딩해줌

all_data['TypeofContact'] = le_TypeofContact.fit_transform(all_data['TypeofContact']) # TypeofContact 컬럼을 인코딩해줌
all_data['Occupation'] = le_Occupation.fit_transform(all_data['Occupation']) # Occupation 컬럼을 인코딩해줌
all_data['Gender'] = le_gender.fit_transform(all_data['Gender']) # Occupation 컬럼을 인코딩해줌
all_data['ProductPitched'] = le_ProductPitched.fit_transform(all_data['ProductPitched']) # Occupation 컬럼을 인코딩해줌
all_data['MaritalStatus'] = le_MaritalStatus.fit_transform(all_data['MaritalStatus']) # Occupation 컬럼을 인코딩해줌
all_data['Designation'] = le_Designation.fit_transform(all_data['Designation']) # Occupation 컬럼을 인코딩해줌




# all_data concat 분리
x = all_data.iloc[:1955, :] # 컬럼을 제외한 모든 컬럼을 x로 저장
test_set_ = all_data.iloc[1955:, :] # 컬럼을 제외한 모든 컬럼을 x로 저장

print(x)
print(test_set)

print(x.shape) # (1956, 18)
print(test_set.shape) # (2932, 18)

# train_set = pd.concat([x, y], axis=1) # axis=0은 행을 더하는 것이라는 뜻

# train_set = np.array(train_set) # numpy array로 변환하기 위해 np.array()함수 사용
# test_set = np.array(test_set) # numpy array로 변환하기 위해 np.array()함수 사용
print(test_set)

#### 결측치 처리 knn 임퓨터 ####
imputer = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3
imputer.fit(x) # 훈련용 데이터로 학습하기 위해 fit()함수 사용
x = imputer.transform(x) # 학습한 데이터로 훈련용 데이터를 이용해서 변환하기 위해 transform()함수 사용
test_set_ = imputer.transform(test_set_) # 학습한 데이터로 훈련용 데이터를 이용해서 변환하기 위해 transform()함수 사용

print(x)
print(y)
print(test_set)



print(x.shape) # (1459, 10)


print(x.shape) # (1956, 9)
print(test_set.shape) # (715, 9)


# x = train_set.drop(['ProdTaken'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.shape) # (1955, 18)

# y = train_set['ProdTaken'] 
y = np.array(y) # numpy array로 변환하기 위해 np.array()함수 사용
print(x)
print(y)
print(test_set_)

# 스케일러, LDA
# scaler = StandardScaler() # 스케일러 적용하기 위해 StandardScaler() 객체 생성
# scaler.fit(x) # 훈련용 데이터로 학습하기 위해 fit()함수 사용
# x = scaler.transform(x) # 학습한 데이터로 훈련용 데이터를 이용해서 변환하기 위해 transform()함수 사용
# test_set = scaler.transform(test_set_) # 학습한 데이터로 훈련용 데이터를 이용해서 변환하기 위해 transform()함수 사용

# lda = LDA() # LDA 객체 생성
# lda.fit(x, y) # 훈련용 데이터로 학습하기 위해 fit()함수 사용
# x = lda.transform(x) # 학습한 데이터로 훈련용 데이터를 이용해서 변환하기 위해 transform()함수 사용
# test_set = lda.transform(test_set) # 학습한 데이터로 훈련용 데이터를 이용해서 변환하기 위해 transform()함수 사용


x_train_val, x_test, y_train_val, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=123, stratify=y
                                                    )
print(x_train_val.shape) # (1564, 18)
print(x_test.shape) # (391, 18)
print(y_train_val.shape) # (1564,)
print(y_test.shape) # (391,)

x_train, x_val, y_train, y_val = train_test_split(x_train_val,
                                                    y_train_val,
                                                    train_size=0.7,
                                                    random_state=123, stratify=y_train_val
                                                    )

print(x_train) # (1298, 18)
print(x_val) # (391, 18)
print(y_train) # (1298,)
print(y_val) # (391,)
print(x_test) # (391, 18)
print(y_test) # (391,)

#2. 모델구성

from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier # xgboost 사용
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline, Pipeline # pipeline을 사용하기 위한 함수

# pipe = Pipeline([('minmax', MinMaxScaler()), ('RFR', RandomForestRegressor())], verbose=1)
# pipe = make_pipeline(MinMaxScaler(), XGBRegressor())
model = GridSearchCV(XGBClassifier(), parameters_xgb,verbose=1,cv=kfold,
                     refit=True,n_jobs=-1) # GridSearchCV를 사용하기 위한 함수
fit_params = {'eval_set': [(x_val, y_val)], 'early_stopping_rounds': 100, 'eval_metric': 'logloss'} # GridSearchCV의 fit_params를 사용하기 위한 함수

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train, y_train, **fit_params) 
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

pred = model.best_estimator_.predict(test_set)
y_summit = [1 if x > 0.5 else 0 for x in pred]

print(y_summit)
# y_summit = y_summit.round()
# df = pd.DataFrame(y_summit)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y_summit = oh.fit_transform(df)
print(y_summit)
# y_summit = np.argmax(y_summit, axis= 1)
submission_set = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

print(submission_set)

submission_set['ProdTaken'] = y_summit
print(submission_set)


submission_set.to_csv(path + 'sample_submission.csv', index = True)

# 최적의 파라미터 : {'classifier__gamma': 0, 'classifier__learning_rate': 0.3, 'classifier__max_depth': 5, 'classifier__min_child_weight': 0.1, 'classifier__n_estimators': 100}
# best_score : 0.8518914163629508
# model_score : 0.8644501278772379
# accuracy_score : 0.8644501278772379
# 최적 튠  ACC : 0.8644501278772379
# 걸린 시간 : 405.91 초

# 최적의 파라미터 : {'gamma': 4, 'learning_rate': 0.1, 'max_depth': 2, 'min_child_weight': 0, 
# 'n_estimators': 100}
# best_score : 0.8427589962716266
# model_score : 0.8414322250639387
# accuracy_score : 0.8414322250639387
# 최적 튠  ACC : 0.8414322250639387
# 걸린 시간 : 199.89 초