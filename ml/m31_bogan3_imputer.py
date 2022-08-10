import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10], 
                     [2, 4, np.nan, 8, np.nan], 
                     [2, 4, 6, 8, 10], 
                     [np.nan, 4, np.nan, 8, np.nan]])

# # print(data)
# print(data.shape) # (4, 5)

data = data.transpose() # (5, 4)
data.columns = ['x1', 'x2', 'x3', 'x4'] # 컬럼 이름 지정
print(data)

from sklearn.experimental import enable_iterative_imputer # 이터러블 입력시 사용하는 모듈 추가
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer # simpleimputer : 값이 없는 곳을 임의의 값으로 채워주는 것, knnimputer : 가장 가까운 값을 채워주는 것, iterativeimputer : 여러 개의 값을 채워주는 것

######################simpleimputer######################
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # missing_values default값은 np.nan, strategy default값은 mean
# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # most_frequent는 가장 많이 나온 값을 채워줌
# imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=777) # constant는 값을 채워주고 싶은 값을 입력해줌
######################knnimputer######################
# imputer = KNNImputer(missing_values=np.nan, n_neighbors=3) # n_neighbors default값은 3
######################IterativeImputer######################
imputer = IterativeImputer(missing_values=np.nan, max_iter=10, tol=0.001) # max_iter default값은 10, tol default값은 0.001


imputer.fit(data) # 데이터프레임에 적용하기 위해 fit()함수 사용
data2 = imputer.transform(data) # transform()함수를 사용하여 데이터프레임을 적용하기 위해 transform()함수 사용
print(data2)
