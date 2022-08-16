import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import sklearn as sk
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__) # 1.1.1

#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (178, 13) (178,)
print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts()) 
# 1    71
# 0    59
# 2    48

print(y)

x = x[:-25] # x의 마지막 40개를 제외한 것
y = y[:-25]

print(pd.Series(y).value_counts()) 
# 1    71
# 0    59
# 2     8


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1004, shuffle=True, stratify=y)
print(pd.Series(y_train).value_counts()) 
# 1    57
# 0    47
# 2     6

#2. 모델
model = RandomForestClassifier()
#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('score : ', score)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

print('f1_score(macro) : ', f1_score(y_test, y_pred, average='macro'))
print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))

# acc :  0.9722222222222222
# f1_score(macro) :  0.9717813051146384

# acc :  0.9642857142857143
# f1_score(macro) :  0.9209876543209877

print('===========SMOTE 적용 후===============')

smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())

model = RandomForestClassifier()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('score : ', score)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
print('f1_score(macro) : ', f1_score(y_test, y_pred, average='macro'))
print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))

