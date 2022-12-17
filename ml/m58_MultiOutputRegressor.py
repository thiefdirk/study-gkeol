import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

#1. 데이터
datasets = load_linnerud()
x, y = datasets.data, datasets.target

#2. 모델


model = Ridge()
model.fit(x, y)
print(model.predict([[2, 110, 43]])) # [[187.32842123  37.0873515   55.40215097]]

model = XGBRegressor()
model.fit(x, y)
print(model.predict([[2, 110, 43]]))# [[138.00215   33.001656  67.99831 ]]

model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
model.fit(x, y)
print(model.predict([[2, 110, 43]]))# [[138.97756017  33.09066774  67.61547996]]

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
print(model.predict([[2, 110, 43]])) # [[178.6  35.4  56.1]]

