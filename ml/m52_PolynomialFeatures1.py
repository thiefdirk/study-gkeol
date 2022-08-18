import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=1234)

#2. 모델구성
model = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=3),
                      LinearRegression())

model.fit(x_train, y_train)

print('model.score : ', model.score(x_test, y_test)) 
# model.score :  0.7665382927362878

# model.score :  0.8742441694450847