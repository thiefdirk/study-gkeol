import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names)
x = datasets['data']
y= datasets['target']

df = pd.DataFrame(x, columns=datasets.feature_names) # 데이터프레임으로 변환

print(df)
df['Target(Y)'] = y # 컬럼 추가
print(df) #[150 rows x 5 columns]


print('========================상관계수 히트 맵============================')
print(df.corr()) # 상관계수 계산

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(df.corr(), annot=True, square=True, cbar=True) # 상관계수 히트맵 출력, annot=True : 상관계수 값 출력, cmap='RdYlGn' : 색상 지정,
                                                           # square=True : 정사각형으로 출력, cbar=True : 상관계수 값 출력

plt.show() # 출력

print('===================================================================')

