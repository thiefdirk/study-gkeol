# 아웃라이어 확인

# 아웃라이어 처리

# 돌려봐!!

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


path = 'D:\study_data\_data/'

dataset = pd.read_csv(path + 'winequality-white.csv', sep=';',
                      index_col=None, header=0) # ;로 구분되어있음


print(dataset.shape) # (4898, 12)
print(dataset.describe())
print(dataset.info())

#################### 그래프 그려봐!! ####################
# 1. value_counts() 사용하지마
# 2. groupby() 사용, count() 사용

# plt.bar 로 그린다(quality 컬럼)

# groupby : quality 컬럼으로 그룹화 
count_data = dataset.groupby('quality')['quality'].count() # quality 컬럼의 값들을 묶어서 각각의 개수를 세줌
# .count()를 하면 quality 컬럼을 제외한 나머지 컬럼들의 값들이 count된다
print(count_data)
plt.bar(count_data.index, count_data) # x축에는 count_data.index, y축에는 count_data
plt.show()
# plt.bar(뭐시기 뭐시기)
# plt.show()


# x = dataset.to_numpy()[:, :-1] # numpy로 바꾸고, 마지막 열을 제외하고 x에 저장
# y = dataset.to_numpy()[:, -1]
# # x = dataset.values[:, :-1] # values는 numpy로 바꿔줌
# # y = dataset.values[:, -1]
# print(dataset.shape) # (4898, 12)
# print(x) # (4898, 11)
# print(y) # (4898,)
# print(type(x)) # <class 'numpy.ndarray'>
# print(type(y)) # <class 'numpy.ndarray'>
# print(np.unique(y, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]

# def outlier(data_out) : 
#     quartile_1, q2, quartile_3 = np.percentile(data_out,
#                                                [25, 50, 75]) # 25%와 75%의 사분위수를 구함, np.percentile()는 정렬된 데이터를 입력받아 사분위수를 구함
#     print('1사분위수 : ', quartile_1)
#     print('50%사분위수 : ', q2)
#     print('3사분위수 : ', quartile_3)
#     iqr = quartile_3 - quartile_1 # 사분위수를 구함
#     print('IQR : ', iqr)
#     lower_bound = quartile_1 - (iqr * 1.5) # 1.5배 사분위수를 구함
#     upper_bound = quartile_3 + (iqr * 1.5) # 1.5배 사분위수를 구함
#     print('최소값 : ', lower_bound)
#     print('최대값 : ', upper_bound)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound)) # 최소값과 최대값 이상의 값을 찾아서 반환함

# outliers_loc = outlier(y) # 최소값과 최대값 이상의 값을 찾아서 반환함
# print('최소값과 최대값 이상의 값을 찾아서 반환함 : ', outliers_loc)
# print(len(outliers_loc[0])) # 200

# x = np.delete(x, outliers_loc, 0) # outliers_loc의 위치에 있는 값을 삭제함
# y = np.delete(y, outliers_loc, 0) # outliers_loc의 위치에 있는 값을 삭제함



# print(np.unique(y, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]
# print(dataset['quality'].value_counts()) # 데이터프레임에서 value_counts() 사용 가능
# # le = LabelEncoder()
# # y = le.fit_transform(y)
# # print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,
#                                                     shuffle=True, stratify=y)


# #2. 모델
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 이진분류모델]


# model = CatBoostClassifier(random_state=12)

# #3. 컴파일, 훈련

# model.fit(x_train, y_train)

# #4. 평가, 예측
# result = model.score(x_test, y_test)
# print('model.score : ', result)
# y_pred = model.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('f1_score(macro) : ', f1_score(y_test, y_pred, average='macro'))
# print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))

# # 이상치 제거 전
# # model.score :  0.7306122448979592
# # 이상치 제거 후
# # model.score :  0.7393617021276596

# #######과제 f1_score presicion recall 정리#########