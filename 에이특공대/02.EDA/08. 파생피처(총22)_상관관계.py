import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# try_num = 

data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)

data = data.reset_index(drop=True)
out_data = out_data.reset_index(drop=True)
test_out_data = test_out_data.reset_index(drop=True)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]
test_out_data = test_out_data.drop(index)
test_out_data_np = test_out_data.to_numpy()

test_sets_path = 'C:/Users/kogas/Desktop/ai_factory/_data/dataset/test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)


for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)



test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
test_sets['year'] = test_sets['TIME'].dt.year
test_sets['month'] = test_sets['TIME'].dt.month
test_sets['day'] = test_sets['TIME'].dt.day
test_sets['hour'] = test_sets['TIME'].dt.hour
test_sets['weekday'] = test_sets['TIME'].dt.weekday
pred_index = test_sets['TIME'].values
test_sets = test_sets.drop(['TIME'], axis=1, inplace=False)

data['bog']=data['FY_SUM'] + data['FIA_SUM']
data['lp_ti'] = data['FY_SUM'] + data['LP_TOTAL']
test_sets['bog']=test_sets['FY_SUM'] + test_sets['FIA_SUM']
test_sets['lp_ti'] = test_sets['FY_SUM'] + test_sets['LP_TOTAL']

x = data
x = data.drop(['PIA205B-02A_MIN', 'PIA205B-02A_MAX'], axis=1)
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y
x = pd.concat([x, out_data], axis=1)


# data['bog']=data['FY_SUM'] + data['FIA_SUM']
# data['lp_ti'] = data['FY_SUM'] + data['LP_TOTAL']
# test_sets['bog']=test_sets['FY_SUM'] + test_sets['FIA_SUM']
# test_sets['lp_ti'] = test_sets['FY_SUM'] + test_sets['LP_TOTAL']

df_corr = x.corr()
df_corr = df_corr.apply(lambda x: round(x,2))
# print(df_corr)

ax = sns.heatmap(df_corr, annot=True, linewidth=.5)
plt.show()

'''
            ZIH120-02  PRESSURE-S  FI_SUM  FY_SUM  TI_MEAN  LP_TOTAL  FIA_SUM  STN-MFR-S  year  month   day  hour  weekday   bog  lp_ti  temp   hum   sun  ground  wind  rain  snow
ZIH120-02        1.00        0.03    0.05    0.02    -0.10      0.05     0.09       0.02 -0.04   0.04 -0.04  0.01    -0.03  0.05   0.05 -0.05 -0.00 -0.09   -0.08 -0.07 -0.03 -0.02
PRESSURE-S       0.03        1.00   -0.14   -0.12    -0.25      0.20    -0.05       0.20  0.02  -0.01  0.00 -0.01     0.02 -0.13   0.20 -0.28 -0.14 -0.01   -0.25 -0.03 -0.05  0.03
FI_SUM           0.05       -0.14    1.00    0.88     0.39     -0.10     0.29      -0.09 -0.07   0.20 -0.02  0.05    -0.23  0.96  -0.06  0.36  0.14  0.04    0.32  0.01  0.04 -0.04
FY_SUM           0.02       -0.12    0.88    1.00     0.34     -0.01    -0.12      -0.00 -0.02   0.18 -0.00  0.04    -0.26  0.94   0.03  0.31  0.13  0.04    0.27  0.03  0.04 -0.04
TI_MEAN         -0.10       -0.25    0.39    0.34     1.00     -0.86     0.08      -0.87 -0.11   0.35 -0.03 -0.01     0.09  0.37  -0.84  0.76  0.49 -0.02    0.67 -0.09  0.06 -0.08
LP_TOTAL         0.05        0.20   -0.10   -0.01    -0.86      1.00    -0.14       0.97  0.10  -0.25  0.01  0.03    -0.21 -0.06   1.00 -0.68 -0.48  0.04   -0.60  0.11 -0.05  0.05
FIA_SUM          0.09       -0.05    0.29   -0.12     0.08     -0.14     1.00      -0.14  0.02  -0.01 -0.06  0.03     0.05  0.22  -0.15  0.11  0.03  0.03    0.11 -0.00  0.01 -0.01
STN-MFR-S        0.02        0.20   -0.09   -0.00    -0.87      0.97    -0.14       1.00  0.09  -0.29  0.02  0.03    -0.22 -0.05   0.97 -0.68 -0.48  0.04   -0.60  0.12 -0.05  0.06
year            -0.04        0.02   -0.07   -0.02    -0.11      0.10     0.02       0.09  1.00  -0.16  0.00 -0.00     0.01 -0.01   0.09 -0.18 -0.07 -0.00   -0.12  0.06 -0.02  0.01
month            0.04       -0.01    0.20    0.18     0.35     -0.25    -0.01      -0.29 -0.16   1.00  0.01  0.00    -0.01  0.17  -0.24  0.31  0.18 -0.04    0.23 -0.10  0.04 -0.08
day             -0.04        0.00   -0.02   -0.00    -0.03      0.01    -0.06       0.02  0.00   0.01  1.00 -0.00     0.00 -0.02   0.01  0.01  0.08 -0.02    0.02  0.00 -0.02 -0.03
hour             0.01       -0.01    0.05    0.04    -0.01      0.03     0.03       0.03 -0.00   0.00 -0.00  1.00    -0.00  0.05   0.03  0.09 -0.04  0.11    0.13  0.04 -0.03 -0.02
weekday         -0.03        0.02   -0.23   -0.26     0.09     -0.21     0.05      -0.22  0.01  -0.01  0.00 -0.00     1.00 -0.24  -0.22 -0.01  0.06 -0.03   -0.00 -0.03 -0.01  0.04
bog              0.05       -0.13    0.96    0.94     0.37     -0.06     0.22      -0.05 -0.01   0.17 -0.02  0.05    -0.24  1.00  -0.02  0.34  0.14  0.04    0.30  0.03  0.04 -0.04
lp_ti            0.05        0.20   -0.06    0.03    -0.84      1.00    -0.15       0.97  0.09  -0.24  0.01  0.03    -0.22 -0.02   1.00 -0.66 -0.47  0.04   -0.59  0.11 -0.05  0.05
temp            -0.05       -0.28    0.36    0.31     0.76     -0.68     0.11      -0.68 -0.18   0.31  0.01  0.09    -0.01  0.34  -0.66  1.00  0.43  0.14    0.89 -0.12  0.05 -0.10
hum             -0.00       -0.14    0.14    0.13     0.49     -0.48     0.03      -0.48 -0.07   0.18  0.08 -0.04     0.06  0.14  -0.47  0.43  1.00 -0.33    0.36 -0.25  0.17  0.02
sun             -0.09       -0.01    0.04    0.04    -0.02      0.04     0.03       0.04 -0.00  -0.04 -0.02  0.11    -0.03  0.04   0.04  0.14 -0.33  1.00    0.39  0.20 -0.08 -0.03
ground          -0.08       -0.25    0.32    0.27     0.67     -0.60     0.11      -0.60 -0.12   0.23  0.02  0.13    -0.00  0.30  -0.59  0.89  0.36  0.39    1.00 -0.01  0.03 -0.07
wind            -0.07       -0.03    0.01    0.03    -0.09      0.11    -0.00       0.12  0.06  -0.10  0.00  0.04    -0.03  0.03   0.11 -0.12 -0.25  0.20   -0.01  1.00  0.10 -0.01
rain            -0.03       -0.05    0.04    0.04     0.06     -0.05     0.01      -0.05 -0.02   0.04 -0.02 -0.03    -0.01  0.04  -0.05  0.05  0.17 -0.08    0.03  0.10  1.00  0.02
snow            -0.02        0.03   -0.04   -0.04    -0.08      0.05    -0.01       0.06  0.01  -0.08 -0.03 -0.02     0.04 -0.04   0.05 -0.10  0.02 -0.03   -0.07 -0.01  0.02  1.00


'''