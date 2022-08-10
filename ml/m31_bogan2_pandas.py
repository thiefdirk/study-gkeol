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

print(data.isnull()) # 열별 여부 확인
print(data.isnull().sum()) # 열별 여부 확인
print(data.info()) # 정보 출력

#1. 결측치 삭제
print('========================결측치 삭제==========================')
print(data.dropna()) # 결측치 삭제
print(data.dropna(axis=1)) # 행별 삭제
print(data.dropna(axis=0)) # 열별 삭제

#2-1. 특정값 - 평균
print('========================결측치 처리 mean()==========================')
mean = data.mean() # 평균값 출력
print('평균값 : ', mean)
data2 = data.fillna(mean) # 평균값으로 채워주기
print('평균값으로 채워주기 : ', data2)

#2-2. 특정값 - 중위값
print('========================결측치 처리 median()==========================')
median = data.median() # 평균값 출력
print('중위값 : ', median)
data3 = data.fillna(median) # 평균값으로 채워주기
print('중위값으로 채워주기 : ', data3)

#2-3. 특정값 - ffill, bfill
print('========================결측치 처리 ffill, bfill==========================')
data4 = data.fillna(method='ffill') # 앞에서 입력된 값을 채워주기 : ffill
print('앞에서 입력된 값을 채워주기 : ffill : ', data4)
data5 = data.fillna(method='bfill') # 뒤에서 입력된 값을 채워주기 : bfill
print('뒤에서 입력된 값을 채워주기 : bfill : ', data5)

#2-4. 특정값 - 임의값
print('========================결측치 처리 임의값==========================')
# data6 = data.fillna(np.random.randint(0, 10)) # 임의값 채워주기, 0~10 사이의 임의값 출력, random.randint(a, b) : a~b 사이의 임의값 출력
data6 = data.fillna(value = 77777)

print('========================특정 컬럼 채우기==========================')

means = data['x1'].mean() # 평균값 출력
print('평균값 : ', means)
data['x1'] = data['x1'].fillna(means) # 평균값으로 채워주기
print('특정 컬럼 평균값으로 채워주기 : ', data)

meds = data['x2'].median() # 중위값 출력
print('중위값 : ', meds)
data['x2'] = data['x2'].fillna(meds) # 중위값으로 채워주기
print('특정 컬럼 중위값으로 채워주기 : ', data)

data['x4'] = data['x4'].fillna(77777)
print('특정 컬럼 임의값으로 채워주기 : ', data)