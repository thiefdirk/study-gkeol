import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__) # 1.1.1
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

pca = PCA(n_components=30) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = pca.fit_transform(x) # x를 pca로 변환한다.
print(x.shape) # (569, 10)

pca_EVR = pca.explained_variance_ratio_ # 주요하지 않은 변수의 중요도를 확인한다.
print(pca_EVR)
print(sum(pca_EVR)) # 1.0

cumsum = np.cumsum(pca_EVR) # 중요도를 이용해 주요하지 않은 변수를 제거한다.
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

# #2. 모델
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor

# # model = XGBRegressor()
# model = RandomForestRegressor()

# #3. 훈련
# model.fit(x_train, y_train) #, eval_metric='error')

# #4. 평가 및 예측
# result = model.score(x_test, y_test)
# print('model.score : ', result)
