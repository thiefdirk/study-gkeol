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
from imblearn.over_sampling import SMOTE



path = 'D:\study_data\_data/'

dataset = pd.read_csv(path + 'winequality-white.csv', sep=';',
                      index_col=None, header=0) # ;로 구분되어있음

print(dataset.shape) # (4898, 12)
print(dataset.describe())
print(dataset.info())

x = dataset.to_numpy()[:, :-1] # numpy로 바꾸고, 마지막 열을 제외하고 x에 저장
y = dataset.to_numpy()[:, -1]
# x = dataset.values[:, :-1] # values는 numpy로 바꿔줌
# y = dataset.values[:, -1]
print(dataset.shape) # (4898, 12)
print(x) # (4898, 11)
print(y) # (4898,)
print(type(x)) # <class 'numpy.ndarray'>
print(type(y)) # <class 'numpy.ndarray'>

print(np.unique(y, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]
print(dataset['quality'].value_counts()) # 데이터프레임에서 value_counts() 사용 가능

newlist = []

# for (index, value) in enumerate(y):
#     if value == 9:
#         y[index] = 7
#     elif value == 8:
#         y[index] = 7
#     elif value == 7:
#         y[index] = 7
#     elif value == 6:
#         y[index] = 6
#     elif value == 5:
#         y[index] = 5
#     elif value == 4:
#         y[index] = 4
#     elif value == 3:
#         y[index] = 4
#     else:
#         y[index] = 0




print(np.unique(y, return_counts=True)) # [4., 5., 6., 7.] [ 183, 1457, 2198, 1060]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,
                                                    shuffle=True, stratify=y)

smote = SMOTE(random_state=123, k_neighbors=3)
x_train, y_train = smote.fit_resample(x_train, y_train, )


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 이진분류모델

model = RandomForestClassifier(random_state=12)

#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))

###############라벨축소만 한거#################
# acc :  0.7295918367346939
# f1_score(micro) :  0.7295918367346939
###############smote만 한거####################


##############둘다 한거#########################
# acc :  0.713265306122449
# f1_score(micro) :  0.7132653061224491 