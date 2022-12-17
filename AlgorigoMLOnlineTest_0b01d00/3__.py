# EDA

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score

train_data = pd.read_csv('C:\study\AlgorigoMLOnlineTest_0b01d00\data\data/train.csv', index_col=0)
test_data = pd.read_csv('C:\study\AlgorigoMLOnlineTest_0b01d00\data\data/test.csv', index_col=0)
unlabel_data = pd.read_csv('C:\study\AlgorigoMLOnlineTest_0b01d00\data\data/unlabeled.csv', index_col=0)

# correlation

# corr = train_data.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(corr, annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':8})
# plt.xticks(rotation=50)
# plt.show()

print(train_data.info())
print(test_data.info())
print(unlabel_data.info())

print(train_data.head())

# labelencoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
unlabed_cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']  

for col in cat_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])
    
for col in unlabed_cat_cols:
    unlabel_data[col] = le.fit_transform(unlabel_data[col])
    
# fillna(0)

train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
unlabel_data = unlabel_data.fillna(0)
    
x_train = train_data.drop(['satisfaction'], axis=1)
y_train = train_data['satisfaction']
x_test = test_data.drop(['satisfaction'], axis=1)
y_test = test_data['satisfaction']
    
print(train_data.head())
print(test_data.head())
print(unlabel_data.head())



# fill nan cols with unlabel_data predict

nan_cols = ['Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
x_train_2 = x_train.drop(nan_cols, axis=1)

unlabel_data_x = unlabel_data.drop(nan_cols, axis=1)
unlabel_data_y = unlabel_data[nan_cols]

x_test_2 = x_test.drop(nan_cols, axis=1)
y_test_2 = x_test[nan_cols]

# xgboostclassifier labelencoding

# le2 = LabelEncoder()

# for col in nan_cols:
#     unlabel_data_y[col] = le2.fit_transform(unlabel_data_y[col])
#     y_test_2[col] = le2.fit_transform(y_test_2[col])

#multi classifcation

rf = RandomForestClassifier(n_estimators=100, random_state=1234, verbose=1)
pred_list = []
change_pred_list = []
for col in nan_cols:
    rf.fit(unlabel_data_x, unlabel_data_y[col])
    # pred = rf.predict(x_test_2)
    # pred_list.append(pred)
    change_pred = rf.predict(x_train_2)
    change_pred_list.append(change_pred)
    
# pred_list = np.array(pred_list)
# pred_list = pred_list.T
# pred_list = pd.DataFrame(pred_list, columns=nan_cols)

change_pred_list = np.array(change_pred_list)
change_pred_list = change_pred_list.T
change_pred_list = pd.DataFrame(change_pred_list, columns=nan_cols)

x_train_2 = x_train_2.reset_index(drop=True)
change_pred_list = change_pred_list.reset_index(drop=True)

x_train_2 = pd.concat([x_train_2, change_pred_list], axis=1)


# nan check

print(x_train_2.isnull().sum())
print(y_train.isnull().sum())


# acc = 0

# for col in nan_cols:
#     acc += accuracy_score(y_test_2[col], pred_list[col])

# print('accuracy_score : ', acc/len(nan_cols)) 
# rf accuracy_score :  0.5074863636363636
# xgb accuracy_score :  0.4420818181818182

    




# model


rf2 = XGBClassifier(n_estimators=100, random_state=1234)
rf2.fit(x_train_2, y_train)
pred = rf2.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, pred)) 

# rf accuracy_score :  0.94025
# xgb accuracy_score :  

# plot_importance

from xgboost import plot_importance

plot_importance(rf2)
plt.show()






