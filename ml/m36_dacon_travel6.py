import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer # 이터러블 입력시 사용하는 모듈 추가
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# m36_dacon_travel.py
import numpy as np
import pandas as pd                               
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
from tqdm import tqdm_notebook



#1. 데이터
path = './_data/dacon_travel/'
train = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

test = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)

sample_submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)


print(train.describe())  # DurationOfPitch, MonthlyIncome
print("=============================상관계수 히트 맵==============")
print(train.corr())                    # 상관관계를 확인.  
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=0.6)
sns.heatmap(data=train.corr(),square=True, annot=True, cbar=True) 
plt.show()



for i in [train, test]:
    i['Gender'] = i['Gender'].map({'Male': 0, 'Female': 1, 'Fe Male': 1})


# pandas의 fillna 메소드를 활용하여 NAN 값을 채워니다.
train_nona = train.copy()

# 0 으로 채우는 경우
train_nona.DurationOfPitch = train_nona.DurationOfPitch.fillna(0)
train_nona.PreferredPropertyStar = train_nona.PreferredPropertyStar.fillna(0)
train_nona.Age = train_nona.Age.fillna(0)

# mean 값으로 채우는 경우
mean_cols = ['NumberOfFollowups','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome']
for col in mean_cols:
    train_nona[col] = train_nona[col].fillna(test[col].mean())

# "Unknown"으로 채우는 경우
train_nona.TypeofContact = train_nona.TypeofContact.fillna("Unknown")

# 결과를 확인합니다.
train_nona.isna().sum()

object_columns = train.columns[train.dtypes == 'object']
print('object 칼럼은 다음과 같습니다 : ', list(object_columns))

# 해당 칼럼만 보아서 봅시다
train[object_columns]

from sklearn.preprocessing import LabelEncoder

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결과를 확인합니다.
train_enc
print(train_enc.isna().sum())


# 결측치 처리
# 0 으로 채우는 경우
test.DurationOfPitch = test.DurationOfPitch.fillna(0)
test.PreferredPropertyStar = test.PreferredPropertyStar.fillna(0)
test.Age = test.Age.fillna(0)

# mean 값으로 채우는 경우
mean_cols = ['NumberOfFollowups','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome']
for col in mean_cols:
    test[col] = test[col].fillna(test[col].mean())

# "Unknown"으로 채우는 경우
test.TypeofContact = test.TypeofContact.fillna("Unknown")

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

print(train_enc.isna().sum())
print(test.isna().sum())


# 모델 선언
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression# import bagging
from sklearn.ensemble import BaggingClassifier
# import voting
from sklearn.ensemble import VotingClassifier
import joblib

# parameters_xgb = [
#     {'n_estimators' : [100, 300, 500] ,
#     'learning_rate' : [ 0.2, 0.5, 1, 0.01, 0.001],
#     'max_depth' : [None, 2, 4, 6, 7],
#     'gamma' : [0, 1, 4, 10, 100],
#     'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5],}]

parameters_xgb = [
    {'gamma': [0], 'learning_rate': [0.2], 'max_depth': [7], 'min_child_weight': [0], 'n_estimators': [300]}]

# model = GridSearchCV(RandomForestClassifier(), parameters_rfr, cv=kfold, n_jobs=-1, verbose=1)
# model = XGBClassifier(random_state=72, n_jobs=-1, n_estimators=100, max_depth=5, learning_rate=0.1, colsample_bytree=0.9, subsample=0.9)
# model = BaggingClassifier(base_estimator=XGBClassifier(), n_estimators=100, random_state=1234)
# model = VotingClassifier(estimators=[('xgb', XGBClassifier()), ('cat', CatBoostClassifier()), ('rfc', RandomForestClassifier())], voting='soft')
model = GridSearchCV(XGBClassifier(gpu_id=0, tree_method='gpu_hist', random_state=66), parameters_xgb, n_jobs=-1, verbose=1)
# model = XGBClassifier(random_state=66)


# 스케일링
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

train_enc[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.fit_transform(train_enc[['Age', 'DurationOfPitch', 'MonthlyIncome']])
test[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test[['Age', 'DurationOfPitch', 'MonthlyIncome']])


# 분석할 의미가 없는 칼럼을 제거합니다.
# 상관계수 그래프를 통해 연관성이 적은것과 - 인것을 빼준다.

drop_col = ['TypeofContact', 'NumberOfChildrenVisiting',
                                'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups']

train = train_enc.drop(columns=drop_col) 
test = test.drop(columns=drop_col)
# train = train_enc.drop(columns=['NumberOfChildrenVisiting','NumberOfPersonVisiting'])  
# test = test.drop(columns=['NumberOfChildrenVisiting','NumberOfPersonVisiting'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]
y = y.values.ravel() # 1차원으로 변환

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15,
                                                    random_state=66) #내일은 이거 지우고 돌려보기

# 모델 학습
model.fit(x_train, y_train)

# 모델 예측
score = model.score(x_test, y_test)

# 예측
y_pred = model.predict(test)

print("최적의 매개변수 :",model.best_estimator_)


print("최적의 파라미터 :",model.best_params_)

 
print("best_score :",model.best_score_)

print("model_score :",model.score(x_test,y_test))

joblib.dump(model, path + 'xgb_grid_strd_15_0823.model')
# 예측된 값을 정답파일과 병합
sample_submission['ProdTaken'] = y_pred

# 정답파일 데이터프레임 확인
print(sample_submission)

sample_submission.to_csv(path+'xgb_grid_strd_15_0823.csv',index = True)


# sample_submission_xgb_basic.csv
# model.score :  0.8721227621483376
# accuracy_score : 0.8721227621483376

# sample_submission_cat.csv
# model.score :  0.8644501278772379
# accuracy_score : 0.8644501278772379

# cat grid search
#  {'depth': 6, 'iterations': 100, 'learning_rate': 0.04}
# model.score :  0.8184143222506394
# accuracy_score : 0.8184143222506394

# cat grid search drop columns
# {'depth': 7, 'iterations': 100, 'learning_rate': 0.04}
# model.score :  0.8209718670076727
# accuracy_score : 0.8209718670076727

# cat_new_grid_drop
#  {'depth': 9, 'learning_rate': 0.04, 'n_estimators': 300}
# model.score :  0.8951406649616368
# accuracy_score : 0.8951406649616368

# cat_new_grid_drop_kfold_statify
#   {'depth': 10, 'learning_rate': 0.04, 'n_estimators': 300}
# model.score :  0.8925831202046036
# accuracy_score : 0.8925831202046036

# # cat_new_grid_drop_statify
# {'depth': 10, 'learning_rate': 0.03, 'n_estimators': 300}
#  0.884918489391333
# model.score :  0.8900255754475703
# accuracy_score : 0.8900255754475703

# cat_new_grid_drop_statify_outlier_kfold
# model.score :  0.8666666666666667
# accuracy_score : 0.8666666666666667
#  The best score across ALL searched params:   
#  0.8790710175172327
#  The best parameters across ALL searched params:
#  {'depth': 10, 'learning_rate': 0.04, 'n_estimators': 300}

# cat_new_grid_drop_statify_outlier_kfold_smote
# model.score :  0.8412698412698413
# accuracy_score : 0.8412698412698413
#  The best score across ALL searched params:
#  0.9141795445053191
#  The best parameters across ALL searched params:
#  {'depth': 10, 'learning_rate': 0.04, 'n_estimators': 300} 

# xgb_new_grid_drop_statify_outlier_kfold_smote
# model.score :  0.8317460317460318
# accuracy_score : 0.8317460317460318
#  The best score across ALL searched params:
#  0.908229123160455
#  The best parameters across ALL searched params:
#  {'gamma': 0, 'learning_rate': 0.3, 'max_depth': 6, 
#   'min_child_weight': 1, 'n_estimators': 100, 'subsample': 1}

# cat3_new_grid_drop_statify_kfold
# model.score :  0.8925831202046036
# accuracy_score : 0.8925831202046036
#  The best score across ALL searched params:
#  0.8842856557712787
#   The best parameters across ALL searched params:
#  {'depth': 10, 'learning_rate': 0.04, 'n_estimators': 300}

# cat3_new_grid_drop_statify
# model.score :  0.8925831202046036
# accuracy_score : 0.8925831202046036
#  Results from Grid Search 

#  The best estimator across ALL searched params:
#  <catboost.core.CatBoostClassifier object at 0x0000012B6D0482E0>

#  The best score across ALL searched params:
#  0.8791560102301791

#  The best parameters across ALL searched params:
#  {'depth': 10, 'learning_rate': 0.04, 'n_estimators': 300}

# cat3_new_grid_statify_kfold
# model.score :  0.9002557544757033
# accuracy_score : 0.9002557544757033
#  The best score across ALL searched params:
#  0.883648726140739
#  The best parameters across ALL searched params:
#  {'depth': 10, 'learning_rate': 0.04, 'n_estimators': 300}

# rfg_grid_statify_kfold
# model.score :  0.887468030690537
# accuracy_score : 0.887468030690537
#  The best score across ALL searched params:
#  0.874694847218809
#   The best parameters across ALL searched params:
#  {'max_features': 0.5, 'min_samples_split': 2, 'n_estimators': 50}

# rfc_grid
# 0.9959079283887468
# 실제 스코어 : 0.8806479113

# rfc_grid_kfold
# 0.9933503836317136
# 실제 스코어 : 0.8959931799

# rfc_basic
# 1.0
# 실제 스코어 : 0.8959931799

# rfc_bagging
# 0.9759590792838875
# 실제 스코어 : 0.8806479113

# rfc_bagging100
# 0.9805626598465473
# 실제 스코어 : 0.8806479113

# xgb_bagging100
# 0.9964194373401535
# 실제 스코어 : 0.8900255754

# xgb_basic
# 실제 스코어 : 0.89769

# cat_basic
# 0.9611253196930947
# 실제 스코어 : 0.8866155158

# cat_voting
# 0.9974424552429667
# 실제 스코어 : 0.8959931799

# xgb_poly
# 1.0
# 0.8900255754

# xgb_basic_new_drp_72
# 0.9514066496163683
# 0.8653026428

# xgb_basic47
# 0.9457800511508951
# 0.8712702472


# xgb_grid, traintest 0.05
# 최적의 파라미터 : {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 7, 'min_child_weight': 0, 'n_estimators': 300}
# best_score : 0.9003796771295249
# model_score : 0.9387755102040817


# xgb_grid, traintest 0.1
# 최적의 파라미터 : {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 7, 'min_child_weight': 0, 'n_estimators': 300}
# best_score : 0.8857355607355608
# model_score : 0.9540816326530612

# xgb_grid_minmax_0823
# 최적의 파라미터 : {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 7, 'min_child_weight': 0, 'n_estimators': 300}
# best_score : 0.9030750949192823
# model_score : 0.9489795918367347