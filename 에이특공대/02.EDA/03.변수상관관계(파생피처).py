import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)

data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)
                           
corr_df1 = data.corr()
corr_df1 = corr_df1.apply(lambda x: round(x,2))
print(corr_df1)

'''
                 ZIH120-02  PIA205B-02A_MIN  PIA205B-02A_MAX  ...   day  hour  weekday
ZIH120-02             1.00            -0.13            -0.12  ... -0.04  0.01    -0.03
PIA205B-02A_MIN      -0.13             1.00             1.00  ... -0.08  0.00     0.03
PIA205B-02A_MAX      -0.12             1.00             1.00  ... -0.08  0.01     0.03
PRESSURE-S            0.03            -0.24            -0.24  ...  0.00 -0.01     0.02
FI_SUM                0.05             0.08             0.08  ... -0.02  0.05    -0.23
FY_SUM                0.02             0.00             0.00  ... -0.00  0.04    -0.26
TI_MEAN              -0.10             0.63             0.63  ... -0.03 -0.01     0.09
LP_TOTAL              0.05            -0.59            -0.59  ...  0.01  0.03    -0.21
FIA_SUM               0.09             0.12             0.13  ... -0.06  0.03     0.05
STN-MFR-S             0.02            -0.61            -0.61  ...  0.02  0.03    -0.22
year                 -0.04             0.02             0.02  ...  0.00 -0.00     0.01
month                 0.04             0.10             0.10  ...  0.01  0.00    -0.01
day                  -0.04            -0.08            -0.08  ...  1.00 -0.00     0.00
hour                  0.01             0.00             0.01  ... -0.00  1.00    -0.00
weekday              -0.03             0.03             0.03  ...  0.00 -0.00     1.00

[15 rows x 15 columns]





'''