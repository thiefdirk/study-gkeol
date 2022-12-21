import pstats
from symbol import eval_input
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.models import load_model
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


data = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/train/train.csv', header=0)

test_sets_path = 'C:/Users/kogas/Desktop/ai_factory/_data/dataset/test/'
test_0 = pd.read_csv(test_sets_path + 'test_0.csv', header=0)

for i in range(5507):
    test_set = pd.read_csv(test_sets_path + f'test_{i+1}.csv', header=0)
    test_0 = pd.concat([test_0, test_set], axis=0)

test_sets = test_0

# print(data)
# print(test_0, test_0.shape)



data['TIME'] = pd.to_datetime(data['TIME'])
data['year'] = data['TIME'].dt.year
data['month'] = data['TIME'].dt.month
data['day'] = data['TIME'].dt.day
data['hour'] = data['TIME'].dt.hour
data['weekday'] = data['TIME'].dt.weekday
data = data.drop(['TIME'], axis=1, inplace=False)
print(data)

test_sets['TIME'] = pd.to_datetime(test_sets['TIME'])
test_sets['year'] = test_sets['TIME'].dt.year
test_sets['month'] = test_sets['TIME'].dt.month
test_sets['day'] = test_sets['TIME'].dt.day
test_sets['hour'] = test_sets['TIME'].dt.hour
test_sets['weekday'] = test_sets['TIME'].dt.weekday
pred_index = test_sets['TIME'].values
test_sets = test_sets.drop(['TIME'], axis=1, inplace=False)

# data.info()
# test_sets.info()
# data.astype(float).dtypes

x = data
y = data.loc[:,['PIA205B-02A_MIN', 'PIA205B-02A_MAX']]

print(x.shape)
print(y.shape)

x = x[:-1]
y = y[1:]

print(x)
print(y)

pred_col = y

# x = x.to_numpy()
# y = y.to_numpy()
# test_sets = test_sets.to_numpy()

# def split_x(seq, size):
#     aaa=[]
#     for i in range(len(seq) - size + 1):
#         subset = seq[i : (i + size)]
#         aaa.append([item for item in subset])
#     return np.array(aaa)

# def split_test(seq, size):
#     aaa=[]
#     for i in range(5508):
#         subset = seq[i*size : (i + 1)*size]
#         aaa.append([item for item in subset])
#     return np.array(aaa)

# size = 6 
# try_num = 3

# print(x.shape) (109728, 15)
# print(y.shape) (109728, 2)

# x = split_x(x, size)
# x = x[:-1,:]
# y = y[6:,:]
# test_sets = split_test(test_sets, size)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False, random_state=66)

model = MultiOutputRegressor(XGBRegressor(colsample_bytree= 0.2330712254400591, 
                                          learning_rate= 0.36710442211072336, 
                                          max_depth= int(3.34169417184538), 
                                          n_estimators= int(3026.9103364687353), 
                                          reg_alpha= 0.6779087573971319, 
                                          reg_lambda= 76.65347278223201, 
                                          subsample=0.8836357421558699))



model.fit(x_train, y_train)


pickle.dump(model, open('C:/Users/kogas/Desktop/ai_factory/_save/Model/pkl_18_xgb.pkl', 'wb'))
# model = pickle.load(model, open('C:/Users/kogas/Desktop/ai_factory/_save/Model/pkl_4_xgb.pkl', 'rb'))

loss = model.score(x_test, y_test)

y_predict = model.predict(x_test)


    

score = r2_score(y_test, y_predict)

submission_predict = model.predict(test_sets)

ccc = []

for i in range(5508):
    subset = submission_predict[6*(i+1)-1]
    ccc.append(subset)
    
submission_predict = ccc

print('loss : ', loss)
print('y_predict : ', y_predict)
# print('submission_predict : ', submission_predict)


submission = pd.read_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_sample.csv', header=0, index_col=0)
# print(submission)
submission.iloc[:, 0:] = submission_predict
# submission = pd.DataFrame(submission_predict, columns=pred_col.columns, index=pred_index)

submission.to_csv('C:/Users/kogas/Desktop/ai_factory/_data/dataset/submission_18th_xgb.csv')


# 0.02393