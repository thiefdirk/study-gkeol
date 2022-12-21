import pstats
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sns
import statsmodels.api as sm
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try_num = 73


data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)
out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/out_data.csv', header=0, index_col=0)
test_out_data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/test_out_data.csv', header=0, index_col=0)


test_out_data = test_out_data.drop(test_out_data.columns[3], axis=1)
out_data = out_data.drop(out_data.columns[3], axis=1)


data = data.reset_index(drop=True)
out_data = out_data.reset_index(drop=True)
test_out_data = test_out_data.reset_index(drop=True)

index = test_out_data.index[(test_out_data.index % 7 == 0) | (test_out_data.index % 8 == 0)]
test_out_data = test_out_data.drop(index)



test_out_data_np = test_out_data.to_numpy()

test_out_data = pd.DataFrame(test_out_data_np, columns=['temp', 'hum', 'sun', 'wind', 'rain', 'snow'])



test_sets_path = 'C:/Users/kogas/Desktop/ai_factory/_data/dataset/test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)
# print(test_out_data)



for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

test_sets = test_sets.reset_index(drop=True)


# print(data)
# print(test_0-test_0.shape)

holiday_list = ['2019-2-4', '2019-2-5', '2019-2-6', '2019-9-12', '2019-9-13', '2019-9-14', '2019-1-1', '2019-3-1', '2019-5-5', '2019-5-12', '2019-6-6', '2019-8-15', '2019-10-9', '2019-10-3', '2019-12-25',
                '2020-1-24', '2020-1-25', '2020-1-26', '2020-9-30', '2020-10-1', '2020-10-2', '2020-1-1', '2020-3-1', '2020-5-5', '2020-4-30', '2020-6-6', '2020-8-15', '2020-10-9', '2020-10-3', '2020-12-25',
                '2021-2-11', '2021-2-12', '2021-2-13', '2021-9-20', '2021-9-21', '2021-9-22', '2021-1-1', '2021-3-1', '2021-5-5', '2021-5-19', '2021-6-6', '2021-8-15', '2021-10-9', '2021-10-3', '2021-12-25']
holiday_list = pd.DataFrame(holiday_list, dtype=object)
holiday_list = pd.to_datetime(holiday_list[0], format="%Y-%m-%d")


data['TIME'] = pd.to_datetime(data['TIME'])
data['dayofyear'] = data['TIME'].dt.dayofyear
data['weekofyear'] = data['TIME'].dt.isocalendar().week
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
# data['day'] = data['TIME'].dt.day
# data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data['holiday'] = ""
data['TIME2'] = data['TIME'].dt.date


for i in range(len(data['TIME'])):
    if (data['TIME2'][i] in holiday_list.to_list()) or (data['weekday'][i] == 5 or data['weekday'][i] == 6):
        data['holiday'][i] = 1
    else:
        data['holiday'][i] = 0




test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
test_sets['dayofyear'] = test_sets['TIME'].dt.dayofyear
test_sets['weekofyear'] = test_sets['TIME'].dt.isocalendar().week
test_sets['year'] = test_sets['TIME'].dt.year
test_sets['month'] = test_sets['TIME'].dt.month
# test_sets['day'] = test_sets['TIME'].dt.day
# test_sets['hour'] = test_sets['TIME'].dt.hour
test_sets['weekday'] = test_sets['TIME'].dt.weekday
pred_index = test_sets['TIME'].values
test_sets['holiday'] = ""
test_sets['TIME2'] = test_sets['TIME'].dt.date


for i in range(len(test_sets['TIME'])):
    if (test_sets['TIME2'][i] in holiday_list.to_list()) | (test_sets['weekday'][i] == 5 | test_sets['weekday'][i] == 6) :
        test_sets['holiday'][i] = 1
    else:
        test_sets['holiday'][i] = 0




oh_encord1 = pd.get_dummies(data['year'])

data = pd.concat([data, oh_encord1], axis=1)

oh_encord2 = pd.get_dummies(test_sets['year'])
oh2019test = pd.DataFrame(0,index=range(len(oh_encord2)), columns = {2019}, dtype=np.uint8)
oh2020test = pd.DataFrame(0,index=range(len(oh_encord2)), columns = {2020}, dtype=np.uint8)
oh_test = pd.concat([oh2019test, oh2020test], axis=1)
oh_encord2 = pd.concat([oh_test, oh_encord2], axis=1)
test_sets = pd.concat([test_sets, oh_encord2], axis=1)

data = pd.concat([data, out_data], axis=1)
test_sets = pd.concat([test_sets, test_out_data], axis=1)
# test_sets = np.concatenate([test_sets,test_out_data], axis=1)


data['bog']=data['FY_SUM'] + data['FIA_SUM']
data['lp_ti'] = data['FY_SUM'] + data['LP_TOTAL']

test_sets['bog']=test_sets['FY_SUM'] + test_sets['FIA_SUM']
test_sets['lp_ti'] = test_sets['FY_SUM'] + test_sets['LP_TOTAL']

cate_list = ['year', 'month']
strd_list = ['LP_TOTAL','STN-MFR-S','PRESSURE-S','FI_SUM', 'FY_SUM','wind', 'rain', 'snow', 'lp_ti', 'bog']
strd_list2 = ['FIA_SUM']
norm_list = ['dayofyear', 'weekofyear', 'TI_MEAN', 'PIA205B-02A_MIN', 'PIA205B-02A_MAX', 'temp', 'hum', 'sun']


data = data.drop(['TIME','TIME2', 'weekday','year'], axis=1, inplace=False)

test_sets = test_sets.drop(['TIME','TIME2', 'weekday','year'], axis=1, inplace=False)

data = data.astype(float)
test_sets = test_sets.astype(float)


# data.groupby(['month'])['PIA205B-02A_MIN'].sum().plot(kind='bar', figsize=(18,4))
# plt.title('월별 최소 압력값')
# plt.show()

# data.groupby(['month'])['PIA205B-02A_MAX'].sum().plot(kind='bar', figsize=(18,4))
# plt.title('월별 최대 압력값')
# plt.show()

# ########################################################################################
# data_=data.drop(['LP_TOTAL', 'STN-MFR-S'], axis=1, inplace=False)

# green_diamond = dict(markerfacecolor='r', marker='s')
# plt.boxplot(data_,flierprops=green_diamond)
# plt.title("train data outlier before scaling")
# plt.xticks(range(1,len(data_.columns)+1),data_.columns, fontsize=10, rotation=20)
# plt.show()
# ########################################################################################

# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ss = StandardScaler()
# rbs = RobustScaler()
# mm = MinMaxScaler()

# ss.fit(data[strd_list2])
# data[strd_list2] = ss.transform(data[strd_list2])
# test_sets[strd_list2]= ss.transform(test_sets[strd_list2]) 

# rbs.fit(data[strd_list])
# data[strd_list] = rbs.transform(data[strd_list])
# test_sets[strd_list]= rbs.transform(test_sets[strd_list])

# mm.fit(data[norm_list])
# data[norm_list] = mm.transform(data[norm_list])
# test_sets[norm_list]= mm.transform(test_sets[norm_list]) 

# ########################################################################################

# data_=data.drop(['LP_TOTAL', 'STN-MFR-S'], axis=1, inplace=False)

# green_diamond = dict(markerfacecolor='r', marker='s')
# plt.boxplot(data_,flierprops=green_diamond)
# plt.title("train data outlier after scaling")
# plt.xticks(range(1,len(data_.columns)+1),data_.columns, fontsize=10, rotation=20)
# plt.show()
# ########################################################################################


# x = x.drop(['PIA205B-02A_MIN', 'PIA205B-02A_MAX'], axis=1)   #종속변수
x = data.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','FI_SUM'], axis=1)   #독립변수
test_sets = test_sets.drop(['STN-MFR-S','LP_TOTAL', 'PRESSURE-S','FI_SUM'], axis=1)
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]
pred_col = y
y_min = y.loc[:,['PIA205B-02A_MIN']]
y_max = y.loc[:,['PIA205B-02A_MAX']]


# test_sets = test_sets.append(test_out_data)

def split_x(seq, size):
    aaa=[]
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

def split_test(seq, size):
    aaa=[]
    for i in range(5508):
        subset = seq[i*size : (i + 1)*size]
        aaa.append([item for item in subset])
    return np.array(aaa)

size = 6 

x = x.to_numpy()
y_min = y_min.to_numpy()
y_max = y_max.to_numpy()
test_sets = test_sets.to_numpy()

x = split_x(x, size)
x = x[:-1,:]
y_min = y_min[6:,:]
y_max = y_max[6:,:]

test_sets = split_test(test_sets, size)

x_reshaped = np.reshape(x, (109722,126))
test_sets_reshaped = np.reshape(test_sets, (5508,126))

print(x_reshaped.shape)

xgb1 = XGBRegressor()
xgb2 = XGBRegressor()


xgb1.fit(x_reshaped, y_min)

xgb2.fit(x_reshaped, y_max)


min_predict = xgb1.predict(test_sets_reshaped)

max_predict = xgb2.predict(test_sets_reshaped)


submission = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_sample.csv', header=0, index_col=0)
# print(submission)
submission.iloc[:, 0] = min_predict
submission.iloc[:, 1] = max_predict
# submission = pd.DataFrame(submission_predict, columns=pred_col.columns, index=pred_index)

pickle.dump(xgb1, open(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/{try_num}th_xgb_min.pkl', 'wb'))
pickle.dump(xgb2, open(f'C:/Users/kogas/Desktop/ai_factory/_save/Model/{try_num}th_xgb_max.pkl', 'wb'))


submission.to_csv(f'C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_{try_num}th_xgb.csv')
