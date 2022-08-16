# SelectFromModel : 앙상블 모델을 사용하여 모델을 선택하는 방법

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import time
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.feature_selection import SelectFromModel


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape) # (442, 10)
print(y.shape) # (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=123)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)
    
#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train, early_stopping_rounds=200, 
          eval_set=[(x_train,y_train), (x_test,y_test)], eval_metric='error', verbose=1)
        #   eval_set=[(x_test,y_test)])
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result)

y_predict = model.predict(x_test)

print('r2_score :',r2_score(y_test,y_predict))

print('model.feature_importances_ : ', model.feature_importances_)

# model.feature_importances_ :  [0.0384281  0.04728455 0.2520316  0.08350179 0.03491126 0.06385776
#  0.04579583 0.03318421 0.32447916 0.07652581]

threshold = model.feature_importances_
print('========================')
for thresh in threshold:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape) # (442, 1)
    print(select_x_test.shape) # (119, 1)
    selection_model = XGBRegressor(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)
    selection_model.fit(select_x_train,y_train, verbose=1)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test,y_predict)
    print('thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))
    print('========================')