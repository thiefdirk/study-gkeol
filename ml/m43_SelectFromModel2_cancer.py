# SelectFromModel : 앙상블 모델을 사용하여 모델을 선택하는 방법

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
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
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape) # (569,)
print(x)
print(y)
drop_number = [8, 28, 17, 19, 24, 25, 10, 0, 16, 13, 14, 3, 5, 18, 9, 11]
x = np.delete(x, drop_number, axis=1)
print(x.shape) # (569, 14)
print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=123)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = XGBClassifier(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)
    
#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train, early_stopping_rounds=200, 
          eval_set=[(x_train,y_train), (x_test,y_test)], eval_metric='logloss', verbose=1)
        #   eval_set=[(x_test,y_test)])
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result)

y_predict = model.predict(x_test)

print('accuracy_score :',accuracy_score(y_test,y_predict))

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
    selection_model = XGBClassifier(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)
    selection_model.fit(select_x_train,y_train, verbose=1)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test,y_predict)
    print('thresh=%.3f, n=%d, acc: %.2f%%' %(thresh, select_x_train.shape[1], score*100.0))
    print('========================')

# 리스트에서 최소값 순차적 반환하기


# FI = model.feature_importances_
# FN = datasets.feature_names
# drop_order = []
# drop_columns = []
# break_num = 0

# while break_num < 16:
#         v1 = min(FI)
#         i1 = np.where(model.feature_importances_ == v1)
#         v2 = datasets.feature_names[i1]
#         drop_order.append(i1[0][0])
#         drop_columns.append(v2[0])
#         i2 = np.where(FI==v1)
#         FI = np.delete(FI, i2)        
#         break_num += 1
#         print(drop_order)
#         print(drop_columns)
#         continue




# [8, 27, 16, 17, 21, 21, 9, 0, 13, 10, 10, 2, 3, 9]
# ['mean symmetry', 'worst symmetry', 'concave points error', 
#  'fractal dimension error', 'worst smoothness', 'worst compactness', 'radius error', 'mean radius', 'concavity error', 'area error', 'smoothness error', 'mean area', 'mean compactness', 'symmetry error']

################drop 전###############################
# model.score :  0.9824561403508771
# accuracy_score : 0.9824561403508771

################drop 후###############################
# model.score :  0.9824561403508771
# accuracy_score : 0.9824561403508771