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
# le = LabelEncoder()
# y = le.fit_transform(y)
# print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,
                                                    shuffle=True, stratify=y)

smote = SMOTE(random_state=123, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train, )

print(np.unique(y_train, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]


#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 이진분류모델

model = CatBoostClassifier()

#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('f1_score(macro) : ', f1_score(y_test, y_pred, average='macro'))
print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))

# model.score :  0.7306122448979592

# model.score :  0.6653061224489796
# f1_score(macro) :  0.41838703742249883
# f1_score(micro) :  0.6653061224489796

# model.score :  0.6744897959183673
# f1_score(macro) :  0.43264912522386006
# f1_score(micro) :  0.6744897959183673