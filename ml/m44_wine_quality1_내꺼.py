import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


path = 'D:\study_data\_data/'

dataset = pd.read_csv(path + 'winequality-white.csv', sep=';') # ;로 구분되어있음

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print(x.shape) # (4898, 11)
print(y.shape) # (4898,)

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)
kfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=123)

parameters = {'cat__depth'         : [4,5,6,7,8,9, 10],
              'cat__learning_rate' : [0.01,0.02,0.03,0.04],
              'cat__n_estimators':[100, 200, 300]
                 }

# parameters = [
#     {'xgb__n_estimators': [100, 200, 300,], 'xgb__learning_rate': [0.1, 0.3, 0.001, 0.01], 'xgb__max_depth': [4, 5, 6]},
# ]
# fit_params = {'eval_set': [(x_train,y_train), (x_test,y_test)], 'early_stopping_rounds': 50, 'eval_metric': 'merror'} # GridSearchCV의 fit_params를 사용하기 위한 함수


# pipe = Pipeline([('scaler', MinMaxScaler()), ('cat', CatBoostClassifier())])
# model = GridSearchCV(pipe, cv=kfold,verbose=True,
#                      refit=True,n_jobs=-1,param_grid=parameters)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# model = CatBoostClassifier()
# model = XGBClassifier()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 이진분류모델

model = RandomForestClassifier(random_state=123)

model.fit(x_train, y_train)
          

result = model.score(x_test, y_test)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print('model.score : ', result)
print('accuracy_score : ', accuracy)

# print('최적의 매개변수 : ', model.best_estimator_)
# print('최적의 매개변수 : ', model.best_params_)
# print('최적의 매개변수 : ', model.best_score_)

# hist = model.evals_result()

# xgb
# model.score :  0.7091836734693877
# accuracy_score :  0.7091836734693877

# cat
# model.score :  0.7183673469387755
# accuracy_score :  0.7183673469387755

# 최적의 매개변수 :  Pipeline(steps=[('scaler', 
# StandardScaler()),
#                 ('cat',
#                  <catboost.core.CatBoostClassifier object at 0x000001AFBD30F8E0>)])
# 최적의 매개변수 :  {'cat__n_estimators': 300, 
# 'cat__learning_rate': 0.04, 'cat__depth': 9}  
# 최적의 매개변수 :  0.6174088409310083

# model.score :  0.6224489795918368
# accuracy_score :  0.6224489795918368
# 최적의 매개변수 :  Pipeline(steps=[('scaler', 
# MinMaxScaler()),
#                 ('cat',
#                  <catboost.core.CatBoostClassifier object at 0x000001FD655BBFA0>)])
# 최적의 매개변수 :  {'cat__n_estimators': 200, 
# 'cat__learning_rate': 0.03, 'cat__depth': 10} 
# 최적의 매개변수 :  0.5997947454844007
