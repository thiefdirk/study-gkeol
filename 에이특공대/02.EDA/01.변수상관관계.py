import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)

corr_df1 = data.corr()
corr_df1 = corr_df1.apply(lambda x: round(x,2))
print(corr_df1)
# print(x.isnull().sum())


'''
PIA205B-02A_MIN, PIA205B-02A_MAX은 TI_MEAN(0.63)과 가장 강한 양의 상관관계를 보임
단 판다스 .corr() 메서드는 단순 상관계수만을 출력해주기 때문에 p-value 확인을 위해서 SciPy 라이브러리 활용 필요성 있음


                 ZIH120-02  PIA205B-02A_MIN  PIA205B-02A_MAX  PRESSURE-S  FI_SUM  FY_SUM  TI_MEAN  LP_TOTAL  FIA_SUM  STN-MFR-S
ZIH120-02             1.00            -0.13            -0.12        0.03    0.05    0.02    -0.10      0.05     0.09       0.02
PIA205B-02A_MIN      -0.13             1.00             1.00       -0.24    0.08    0.00     0.63     -0.59     0.12      -0.61
PIA205B-02A_MAX      -0.12             1.00             1.00       -0.24    0.08    0.00     0.63     -0.59     0.13      -0.61
PRESSURE-S            0.03            -0.24            -0.24        1.00   -0.14   -0.12    -0.25      0.20    -0.05       0.20
FI_SUM                0.05             0.08             0.08       -0.14    1.00    0.88     0.39     -0.10     0.29      -0.09
FY_SUM                0.02             0.00             0.00       -0.12    0.88    1.00     0.34     -0.01    -0.12      -0.00
TI_MEAN              -0.10             0.63             0.63       -0.25    0.39    0.34     1.00     -0.86     0.08      -0.87
LP_TOTAL              0.05            -0.59            -0.59        0.20   -0.10   -0.01    -0.86      1.00    -0.14       0.97
FIA_SUM               0.09             0.12             0.13       -0.05    0.29   -0.12     0.08     -0.14     1.00      -0.14
STN-MFR-S             0.02            -0.61            -0.61        0.20   -0.09   -0.00    -0.87      0.97    -0.14       1.00

'''

'''

ZIH120-02          0
PIA205B-02A_MIN    0
PIA205B-02A_MAX    0
PRESSURE-S         0
FI_SUM             0
FY_SUM             0
TI_MEAN            0
LP_TOTAL           0
FIA_SUM            0
STN-MFR-S          0
year               0
month              0
day                0
hour               0
weekday            0
dtype: int64

'''

