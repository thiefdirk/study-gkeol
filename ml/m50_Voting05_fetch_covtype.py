import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer, fetch_california_housing, fetch_covtype, load_wine, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, r2_score


parameters = [
    {'classifier__n_estimators' : [100, 200, 300, 400, 500] ,
    'classifier__learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 1, 0.01, 0.001],
    'classifier__max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
    'classifier__min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100],}]

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)

la = LabelEncoder()
y = la.fit_transform(y)

# pca = PCA(n_components=6) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
# x = pca.fit_transform(x) # x를 pca로 변환한다.
lda = LDA(n_components=5) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = lda.fit_transform(x, y)
# pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.

# cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(np.unique(y_train, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]) -> (array([0, 1, 2, 3, 4, 5, 6])

# 2. 모델
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor

cat = CatBoostClassifier(verbose=0)
lgbm = LGBMClassifier()
xgb = XGBClassifier()

model = VotingClassifier(estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)],
                         voting='soft') # hard : 다수결, soft : 확률값, 통상적으로 soft가 더 좋다.


#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train,)
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

score = accuracy_score(y_test, model.predict(x_test))


classifiers = [cat, lgbm, xgb]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, score2))
    
print('보팅 결과 : ', round(score, 4))


# fetch_covtype
# model.score :  0.8746848188084645
# model.fit 시간 :  6.83866286277771

# fetch_covtype PCA // n_components=6
# model.score :  0.8365360618916895
# model.fit 시간 :  3.9195711612701416

# fetch_covtype LDA // n_components=6
# model.score :  0.7888694784127777
# model.fit 시간 :  3.6569178104400635

