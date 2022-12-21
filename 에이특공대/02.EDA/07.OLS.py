import pstats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]

out_data = out_data.reset_index()

out_data = out_data.drop('index', axis=1)


data = pd.concat([data, out_data], axis=1)

print(data)

data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)

import statsmodels.formula.api as smf
form ='PIA205B-02A_MAX' + '~' + 'FI_SUM' +'+'+ 'FY_SUM'
model = smf.ols(form, data=data).fit()
print(model.summary())




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
temp               0
hum                0
sun                0
ground             0
wind               0
rain               0
snow               0
year               0
month              0
day                0
hour               0
weekday  

'''