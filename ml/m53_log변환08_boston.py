import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # log변환
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=1234)

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

model = RandomForestRegressor()
# model = LinearRegression()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

y_pred = model.predict(x_test)
results = r2_score(y_test, y_pred)
print('r2_score : ', round(results, 4))

# r2_score :  0.7665 -> 0.9127

######################## log변환 #########################

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)



print(df['B'].head())
df['B'] = np.log1p(df['B']) # log1p : log(1+x) 로그변환 결과 : 0.7711
# df['CRIM'] = np.log1p(df['CRIM']) # log1p : log(1+x) 로그변환 결과 :0.7596
df['ZN'] = np.log1p(df['ZN']) # log1p : log(1+x) 로그변환 결과 :0.7734
df['TAX'] = np.log1p(df['TAX']) # log1p : log(1+x) 로그변환 결과 :0.7669
print(df['B'].head())

x_train, x_test, y_train, y_test = train_test_split(df,y,
                                                    test_size=0.2,
                                                    random_state=1234)

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성

model = RandomForestRegressor()
# model = LinearRegression()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

y_pred = model.predict(x_test)
results = r2_score(y_test, y_pred)
print('r2_score : ', round(results, 4))

# r2_score :  0.9141