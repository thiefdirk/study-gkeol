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



#1. 데이터
path = './_data/dacon_autopilot/open/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# 결측치를 처리하는 함수를 작성.
drop_col = ['X_04', 'X_11', 'X_02', 'X_01', 'X_24', 'X_37'] # 컬럼 삭제하기 위한 리스트 생성
train_set = train_set.drop(drop_col, axis=1) # axis=1 : 세로, axis=0 : 가로
test_set = test_set.drop(drop_col, axis=1) # 결측치 처리하기 위한 함수 실행
kfold = KFold(n_splits=5, shuffle=True, random_state=42) # 정규화된 데이터로 교차검증 수행

out_put_col = ['Y_01','Y_02','Y_03','Y_04','Y_05','Y_06','Y_07','Y_08','Y_09','Y_10','Y_11','Y_12','Y_13','Y_14']

# 결측치 확인
print(train_set.isnull().sum()) # 결측치 확인


# 모델 선언
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
# MultiRMSE import

# model = CatBoostRegressor()

# # 분석할 의미가 없는 칼럼을 제거합니다.
# train = train_enc.drop(columns=['TypeofContact','Occupation'])
# test = test.drop(columns=['TypeofContact','Occupation'])


# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x = train_set.drop(out_put_col, axis=1) # 학습에 사용할 정보를 분리합니다.
y = train_set[out_put_col] # 예측하고자 하는 정보를 분리합니다.

# x = np.array(x) # numpy 배열로 변환합니다.
# y = np.array(y) # numpy 배열로 변환합니다.
# test_set = np.array(test_set) # numpy 배열로 변환합니다.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (31685, 56) (7922, 56) (31685, 14) (7922, 14)


# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)




# model = XGBRegressor()
# model = LGBMRegressor()
# model = MultiOutputRegressor(LGBMRegressor())
model = MultiOutputRegressor(CatBoostRegressor())

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

submission_set.to_csv(path +str(score)+ '_cat_drp_MultiOutputRegressor.csv', index = True)

import joblib
joblib.dump(model, path +str(score)+ '_cat_drp_MultiOutputRegressor.model') # 저장하기

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


