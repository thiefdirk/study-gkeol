import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]

out_data = out_data.reset_index()

out_data = out_data.drop('index', axis=1)


data = pd.concat([data, out_data], axis=1)


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

print(data.columns)
# data_=[data['temp'],data['hum'],data['sun'],data['ground'],data['wind'],data['rain'],data['snow'],]
# data=data.drop(['LP_TOTAL', 'STN-MFR-S','year', 'TI_MEAN'], axis=1, inplace=False)

green_diamond = dict(markerfacecolor='r', marker='s')
plt.boxplot(data,flierprops=green_diamond)
plt.title("weather box plot")
plt.xticks(range(1,len(data.columns)+1),data.columns, fontsize=10, rotation=20)
plt.show()


data.plot(kind="scatter", x='temp', y='PIA205B-02A_MIN')
    

cate_list = ['ZIH120-02']
strd_list = ['LP_TOTAL','STN-MFR-S','PRESSURE-S','FI_SUM', 'FY_SUM','FIA_SUM','wind', 'rain', 'snow']
norm_list = ['dayofyear', 'weekofyear', 'TI_MEAN', 'PIA205B-02A_MIN', 'PIA205B-02A_MAX', 'temp', 'hum', 'sun']
all_col = ['ZIH120-02', 'PIA205B-02A_MIN', 'PIA205B-02A_MAX', 'PRESSURE-S',
       'FI_SUM', 'FY_SUM', 'TI_MEAN', 'LP_TOTAL', 'FIA_SUM', 'STN-MFR-S',
       'temp', 'hum', 'sun', 'ground', 'wind', 'rain', 'snow', 'year', 'month',
       'day', 'hour', 'weekday']