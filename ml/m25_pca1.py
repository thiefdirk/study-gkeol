# PCA
# standard scaler 랑 궁합좋음
# 보통 scaler 한방 때리고 pca로 차원 축소해서 작업들어가는 사람들 많음(의견 갈림)

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__) # 1.1.1
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

pca = PCA(n_components=12) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
x = pca.fit_transform(x) # x를 pca로 변환한다.
print(x.shape) # (506, 2)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# model = XGBRegressor()
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train) #, eval_metric='error')

#4. 평가 및 예측
result = model.score(x_test, y_test)
print('model.score : ', result)

# (506, 13)
# model.score :  0.9237116766817286

# (506, 2)
# model.score :  0.46629034183517293

# (506, 3)
# model.score :  0.47649447639232567

# (506, 4)
# model.score :  0.480499681339703

# (506, 8)
# model.score :  0.8112437772246838

# (506, 10)
# model.score :  0.8252626468807018

# (506, 11)
# model.score :  0.9022909944663096
