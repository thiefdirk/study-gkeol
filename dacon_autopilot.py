import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict, GridSearchCV,StratifiedKFold, RandomizedSearchCV
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
from sklearn.multioutput import MultiOutputRegressor
# import pca
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization # pip install bayesian-optimization

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    return score

# parameters = {'depth'         : [4,5,6,7,8],
#               'learning_rate' : [0.01,0.02,0.03],
#               'n_estimators':[100, 200, 300]
#                  }

parameters = {'iterations': [600, 700, 800, 900, 1000],
          'depth': [4, 5, 6],
          'l2_leaf_reg': np.logspace(-20, -19, 3),
          'leaf_estimation_iterations': [10]} # cat_prams


#1. 데이터
path = './_data/dacon_autopilot/open/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# 결측치를 처리하는 함수를 작성.

# train_set.plot.box() # x_49, x_8
# plt.show()

# train_set['X_49'] = np.log1p(train_set['X_49']) # log1p : log(1+x) 로그변환 결과 : 0.7711
# train_set['X_08'] = np.log1p(train_set['X_08']) # log1p : log(1+x) 로그변환 결과 : 0.7711
# test_set['X_49'] = np.log1p(test_set['X_49']) # log1p : log(1+x) 로그변환 결과 : 0.7711
# test_set['X_08'] = np.log1p(test_set['X_08']) # log1p : log(1+x) 로그변환 결과 : 0.7711



drop_col = ['X_04', 'X_11', 'X_02', 'X_01', 'X_24', 'X_37', 'X_23', 'X_47', 'X_48'] # 컬럼 삭제하기 위한 리스트 생성
train_set = train_set.drop(drop_col, axis=1) # axis=1 : 세로, axis=0 : 가로
test_set = test_set.drop(drop_col, axis=1) # 결측치 처리하기 위한 함수 실행
kfold = KFold(n_splits=5, shuffle=True, random_state=42) # 정규화된 데이터로 교차검증 수행

out_put_col = ['Y_01','Y_02','Y_03','Y_04','Y_05','Y_06','Y_07','Y_08','Y_09','Y_10','Y_11','Y_12','Y_13','Y_14']

# 결측치 확인
print(train_set.isnull().sum()) # 결측치 확인
print(train_set.shape)

# 모델 선언
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x = train_set.drop(out_put_col, axis=1) # 학습에 사용할 정보를 분리합니다.
y = train_set[out_put_col] # 예측하고자 하는 정보를 분리합니다.


# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# test_set = scaler.transform(test_set)

# pca = PCA(n_components=46)
# x = pca.fit_transform(x)
# test_set = pca.transform(test_set)
# pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
# cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.
# print('n_components=', 63, ':') # 중요도를 이용해 주요하지 않은 변수를 제거한다.
# print(np.argmax(cumsum >= 0.95)+1) #34
# print(np.argmax(cumsum >= 0.99)+1) #41
# print(np.argmax(cumsum >= 0.999)+1) #46
# print(np.argmax(cumsum+1)) #46


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (31685, 56) (7922, 56) (31685, 14) (7922, 14)

# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)


# bayesian_params = {
#     'max_depth': (6,16),
#     'num_leaves': (24, 64),
#     'min_child_samples': (10, 200),
#     'min_child_weight': (1, 50),
#     'subsample': (0.5, 1),
#     'colsample_bytree': (0.5, 1),
#     'max_bin': (10, 500),
#     'reg_lambda': (0.001, 10),
#     'reg_alpha': (0.01, 50),
# }

# def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,
#             subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
#     params = {'n_estimators': 500, 'learning_rate' : 0.02,
#         'max_depth': int(max_depth),
#         'num_leaves': int(num_leaves),
#         'min_child_samples': int(min_child_samples),
#         'min_child_weight': min_child_weight,
#         'subsample': subsample,
#         'colsample_bytree': colsample_bytree,
#         'max_bin': int(max_bin),
#         'reg_lambda': reg_lambda,
#         'reg_alpha': reg_alpha,
#     }
#     model = LGBMRegressor(**params, 
#                            gpu_id=0, tree_method='gpu_hist',random_state=1234)
#     model.fit(x_train, y_train,
#               eval_set=[(x_train, y_train), (x_test, y_test)],
#               eval_metric='rmse',
#               verbose=100,
#               early_stopping_rounds=100)
#     y_pred = model.predict(x_test)
#     results = lg_nrmse(np.array(y_test), y_pred)
#     return results

# lgb_bo = BayesianOptimization(lgb_hamsu, bayesian_params, random_state=1234)
# lgb_bo.maximize(init_points=5, n_iter=30)
# print(lgb_bo.max)

# exit()

#2. 모델 구성
# model = XGBRegressor()
# model = LGBMRegressor()
# model = MultiOutputRegressor(LGBMRegressor())
# model = CatBoostRegressor(loss_function='MultiRMSE', random_state=123, verbose=1000)
model = GridSearchCV(CatBoostRegressor(loss_function='MultiRMSE', random_state=123, verbose=1000),
                     param_grid=parameters,verbose=1) # GPU 사용
# model = RandomizedSearchCV(CatBoostRegressor(loss_function='MultiRMSE',random_state=123, verbose=1000),
#                            param_distributions=parameters, cv=5, verbose=1) # GPU 사용

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train, y_train)  # **fit_params
end = time.time()- start
#4. 평가, 예측
# result = model.score(x_test, y_test)

# print('model.score : ', result) # model.score :  1.0

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)
print('y_test : ', np.array(y_test))

score = lg_nrmse(np.array(y_test), y_predict)
print('NRMSE : ',score) # 0.0


y_summit = model.predict(test_set)
# y_summit = [1 if x > 0.5 else 0 for x in pred]

submission_set = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

submission_set[out_put_col] = y_summit # 예측값을 예측값으로 채워넣기

submission_set.to_csv(path +str(score)+ '_cat__drp2_123_gridcv.csv', index = True)

import joblib
joblib.dump(model, path +str(score)+ '_cat__ drp2_123_gridcv.model') # 저장하기

print('걸린시간 : ', end)
print('best_params : ', model.best_params_)
print('best_score : ', model.best_score_)
print('best_estimator : ', model.best_estimator_)
print('best_index : ', model.best_index_)



# model.save_model(path +str(score)+  '_MultiOutputRegressor.model') # 모델 저장


# xgb_basic
# NRMSE :  1.9934298355377522
# score : 2.0050576846

# xgb_drop_col
# NRMSE :  1.9934566473103086
# score : 2.0052090378

# xgb_drp_col_minmax
# NRMSE :  3.504365977521119

# xgb_minmax
# NRMSE :  3.280969619390634

# xgb_stnrd
# NRMSE :  1.9978812456185029

# xgb_stnrd_drp
# NRMSE :  1.9963029946395503
# score : 4.9596482854

# MultiOutputRegressor(XGBRegressor)
# NRMSE :  1.9934298355377522
# 2.0050576846

# MultiOutputRegressor(LGBMRegressor)
# NRMSE :  1.9389336868686062
# 1.9530833905

# MultiOutputRegressor(catboostregressor)
# NRMSE :  1.9376827157308794
# 1.9498334591

# MultiOutputRegressor(LGBMRegressor)_drop_col
# NRMSE :  1.941478542140069
# 1.9531576506

# MultiOutputRegressor(catboostregressor)_drop_col
# NRMSE :  1.9382488450118323
# 1.9514119887

# _cat_drp_
# NRMSE :  1.940710356884139
# 1.9477700047

# _cat_drp_stndscl
# NRMSE :  1.9408016383496292
# 2.0844623159

# _cat_drp_minmax
# NRMSE :  1.9993093980815655

# _cat_drp_robust
# NRMSE :  1.9400422657496188
# 2.0586802241	

# _cat
# NRMSE :  1.9425595007225172
# 1.9492589777

# _cat_stnd_PCA(n_components=10)
# NRMSE :  1.9687503991716546
# 1.9721989374

# _cat_stnd_PCA46
# NRMSE :  1.962560734583894
# 1.9660133015

# _cat_drp_123
# NRMSE :  1.9407315246724302
# 1.9464067607

# _cat_drp2_123
# NRMSE :  1.9407315246724302
# 1.9464067607

# _cat_drp2_123_log
# NRMSE :  1.9407315083432233
# 1.9464068161

# _cat_drp2_123_log2
# NRMSE :  1.9407316175665534
# 1.9464068161

# _cat_drp2_123_randcv_kfold
# NRMSE :  1.9564719071720698
# 걸린시간 :  616.8261032104492
# best_params :  {'n_estimators': 200, 'learning_rate': 0.03, 'depth': 7}

# _cat_drp2_123_gridcv
# NRMSE :  1.9412497852996515
# 걸린시간 :  7632.0811603069305
# best_params :  {'depth': 6, 'iterations': 1000,
#                 'l2_leaf_reg': 1e-20, 'leaf_estimation_iterations': 10}  



# threshold = model.feature_importances_
# print('========================')
# for thresh in threshold:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape) # (442, 1)
#     print(select_x_test.shape) # (119, 1)
#     selection_model = XGBRegressor()
#     selection_model.fit(select_x_train,y_train, verbose=1)
#     y_predict = selection_model.predict(select_x_test)
#     score = lg_nrmse(np.array(y_test),y_predict)
#     print('thresh=%.3f, n=%d, nrmse: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))
#     print('========================')
    
# (31685, 50)
# (7922, 50)

# FI = model.feature_importances_
# FN = train_set.columns # 컬럼명
# drop_order = []
# drop_columns = []
# break_num = 0

# while break_num < 6:
#         v1 = min(FI)
#         i1 = np.where(model.feature_importances_ == v1)
#         v2 = train_set.columns[i1]
#         drop_order.append(i1[0][0])
#         drop_columns.append(v2[0])
#         i2 = np.where(FI==v1)
#         FI = np.delete(FI, i2)        
#         break_num += 1
#         print(drop_order)
#         print(drop_columns)
#         continue


