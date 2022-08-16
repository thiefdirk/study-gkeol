# smote 넣어서 맹그러
# 넣고 안넣고 비교

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


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target


print(np.unique(y, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]
# le = LabelEncoder()
# y = le.fit_transform(y)
# print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,
                                                    shuffle=True, stratify=y)

# smote = SMOTE(random_state=123, k_neighbors=1)
# x_train, y_train = smote.fit_resample(x_train, y_train, )

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

###################smote 적용 전###################
# model.score :  0.9824561403508771
# f1_score(macro) :  0.981329839502129
###################smote 적용 후###################
# model.score :  0.9824561403508771
# f1_score(macro) :  0.981329839502129

