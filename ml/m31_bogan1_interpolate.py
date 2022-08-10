# 결측치 처리
#1. 행 또는 열 삭제
#2. 임의의 값으로 채워주기
#   mean
#   median
#   0 : fillna(0)
#   앞에서 입력된 값을 채워주기 : ffill
#   뒤에서 입력된 값을 채워주기 : bfill
#   특정 값으로 채워주기 : fillna(value)
#3. 보간법 사용하기 - interpolate() (선형회귀방식으로 찾아냄)
#4. 모델 - predict로 예측하기 ex) model.predict([3])
#5. 부스팅계열(xgboost,randomforest) - score로 평가하기 ex) model.score(x_test, y_test)

import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022', '8/11/2022', '8/12/2022', '8/13/2022', '8/14/2022']

dates = pd.to_datetime(dates)
print(dates)
print('========================================================')
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates) # 판다스는 데이터프레임, 시리즈. 시리즈가 모이면 데이터프레임
print(ts)

print('========================================================')

ts= ts.interpolate() # 선형회귀방식으로 찾아냄
print(ts)
