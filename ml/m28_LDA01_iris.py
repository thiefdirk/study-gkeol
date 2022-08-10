import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer, fetch_california_housing, fetch_covtype, load_wine, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import LabelEncoder

'''
01. iris
02. cancer
03. diabetes
04. wine
05. covtype
06. digit
07. kaggle_titanic
'''

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]) -> (array([0, 1, 2, 3, 4, 5, 6])


# la = LabelEncoder()
# y = la.fit_transform(y)

# pca = PCA(n_components=9) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
# x = pca.fit_transform(x) # x를 pca로 변환한다.
# lda = LDA(n_components=7) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
# x = lda.fit_transform(x, y)
# pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.

# cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

print(np.unique(y_train, return_counts=True)) # (array([1, 2, 3, 4, 5, 6, 7]) -> (array([0, 1, 2, 3, 4, 5, 6])

# 2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

# 3. 훈련
start= time.time()
model.fit(x_train, y_train)
end = time.time() - start
# 4. 평가 및 예측
result = model.score(x_test, y_test)
print('model.score : ', result)
print('model.fit 시간 : ', end) # model.fit 시간 :  0.0012098692321827698

# load_iris
# model.score :  0.9555555555555556
# model.fit 시간 :  0.3517017364501953

# load_iris LDA // n_components : 7
# model.score :  0.9444444444444444
# model.fit 시간 :  0.20922255516052246

# load_iris PCA // n_components : 9
# model.score :  0.9416666666666667
# model.fit 시간 :  0.2821693420410156