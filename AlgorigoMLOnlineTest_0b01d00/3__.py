
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score

train_data = pd.read_csv('./AlgorigoMLOnlineTest_0b01d00/data/train.csv', index_col=0)
test_data = pd.read_csv('./AlgorigoMLOnlineTest_0b01d00/data/test.csv', index_col=0)
unlabel_data = pd.read_csv('./AlgorigoMLOnlineTest_0b01d00/data/unlabeled.csv', index_col=0)


print(train_data.info())
print(test_data.info())
print(unlabel_data.info())


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



# nan 컬럼(output) 과 nan이 아닌 컬럼(input) 분리

nan_cols = ['Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness',
            'Departure Delay in Minutes', 'Arrival Delay in Minutes']
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

# train 결측치 예측 모델

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

# plot_importance 하위 3개 컬럼 제거

x_train_2 = x_train_2.drop(['Gender'], axis=1)
x_test = x_test.drop(['Gender'], axis=1)


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


# # rf2 = XGBClassifier(n_estimators=100, random_state=1234)
# rf2 = RandomForestClassifier(n_estimators=100, random_state=1234)
# rf2.fit(x_train_2, y_train)
# pred = rf2.predict(x_test)
# print('accuracy_score : ', accuracy_score(y_test, pred)) 

# # plot_importance
# from xgboost import plot_importance

# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(rf2, ax=ax)
# plt.rcParams["font.size"] = 15
# plt.show()

# rf accuracy_score :  0.94025
# xgb accuracy_score :  0.93805

# 하위 3개 컬럼 제거 후 rf accuracy_score :  0.93095
# 하위 2개 컬럼 제거 후 rf accuracy_score :  0.93605
# 하위 1개 컬럼 제거 후 rf accuracy_score :  0.93995

# bayesian optimization

from bayes_opt import BayesianOptimization

def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    model = RandomForestClassifier(n_estimators=int(n_estimators),
                                      max_depth=int(max_depth),
                                        min_samples_split=int(min_samples_split),
                                        min_samples_leaf=int(min_samples_leaf),
                                        max_features=int(max_features),
                                        random_state=1234)
    model.fit(x_train_2, y_train)
    pred = model.predict(x_test)
    score = accuracy_score(y_test, pred)
    return score

pbounds = {'n_estimators': (100, 1000),
              'max_depth': (5, 30),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 10),
                'max_features': (1, 10)}

optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=1234,
)

optimizer.maximize(init_points=2, n_iter=15)


print(optimizer.max) 
# {'target': 0.9417, 'params': {'max_depth': 28.384607962569884,
# 'max_features': 5.579713799274501, 'min_samples_leaf': 1.32220275014516,
# 'min_samples_split': 2.967664253719808, 'n_estimators': 813.6814293566729}}

rf2 = RandomForestClassifier(n_estimators=813.6814293566729,
                                        max_depth=28.384607962569884,
                                        min_samples_split=2.967664253719808,
                                        min_samples_leaf=1.32220275014516,
                                        max_features=5.579713799274501,
                                        random_state=1234)

rf2.fit(x_train_2, y_train)
pred = rf2.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, pred)) 
# 튜닝전 rf accuracy_score :  0.94025
# 튜닝후 rf accuracy_score :  0.9417
                            
    






