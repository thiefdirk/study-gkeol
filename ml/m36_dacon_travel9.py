from time import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from tpot import TPOTClassifier


#1. 데이터
path = './_data/dacon_travel/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
print(train_set[train_set['ProdTaken'].notnull()].groupby(['ProductPitched'])['ProdTaken'].mean())
print(train_set['ProductPitched'].value_counts())
print(train_set[train_set['ProdTaken'].notnull()].groupby(['OwnCar'])['ProdTaken'].mean())
print(train_set['OwnCar'].value_counts())
# cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']


train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)



train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
print(train_set.describe) #(1955, 19)
print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())

train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
# combine = [train_set,test_set]
# for dataset in combine:    
#     dataset.loc[ dataset['NumberOfChildrenVisiting'] < 1, 'NumberOfChildrenVisiting'] = 0
#     dataset.loc[ dataset['NumberOfChildrenVisiting'] >= 1, 'NumberOfChildrenVisiting'] = 1
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())
# print(train_set.isnull().sum()) 

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)
# train_set['DurationOfPitch'].fillna(train_set.groupby('NumberOfChildrenVisiting')['DurationOfPitch'].transform('mean'), inplace=True)
# test_set['DurationOfPitch'].fillna(test_set.groupby('NumberOfChildrenVisiting')['DurationOfPitch'].transform('mean'), inplace=True)
# print(train_set.isnull().sum()) 


print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())


train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
# print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['ProdTaken'])['PreferredPropertyStar'].mean())


# train_set['Ageband'] = train_set['Age']
# test_set['Ageband'] = test_set['Age']
# 임의로 5개 그룹을 지정

# [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] <
# (43.8, 52.4] < (52.4, 61.0]]
# print(train_set['Age'].unique())

combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5


train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfChildrenVisiting'].notnull()].groupby(['MaritalStatus'])['NumberOfChildrenVisiting'].mean())

# print(train_set['Occupation'].unique()) # ['Small Business' 'Salaried' 'Large Business' 'Free Lancer']
train_set.loc[ train_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test_set.loc[ test_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

for i in cols:
    le = LabelEncoder()
    train_set[i]=le.fit_transform(train_set[i])
    test_set[i]=le.fit_transform(test_set[i])

# print(train_set)

# print(train_set['TypeofContact'])
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
                     
                           
# print(train_set['Designation'].unique())

# Age_out_index= outliers(train_set['Age'])[0]
# TypeofContact_out_index= outliers(train_set['TypeofContact'])[0] # 0
# CityTier_out_index= outliers(train_set['CityTier'])[0] # 0
DurationOfPitch_out_index= outliers(train_set['DurationOfPitch'])[0] #44
Gender_out_index= outliers(train_set['Gender'])[0] # 0
NumberOfPersonVisiting_out_index= outliers(train_set['NumberOfPersonVisiting'])[0] # 1
NumberOfFollowups_out_index= outliers(train_set['NumberOfFollowups'])[0] # 0
ProductPitched_index= outliers(train_set['ProductPitched'])[0] # 0
PreferredPropertyStar_out_index= outliers(train_set['PreferredPropertyStar'])[0]  # 0
MaritalStatus_out_index= outliers(train_set['MaritalStatus'])[0] # 0
NumberOfTrips_out_index= outliers(train_set['NumberOfTrips'])[0] # 38
Passport_out_index= outliers(train_set['Passport'])[0] # 0
PitchSatisfactionScore_out_index= outliers(train_set['PitchSatisfactionScore'])[0] # 0
OwnCar_out_index= outliers(train_set['OwnCar'])[0] # 0
NumberOfChildrenVisiting_out_index= outliers(train_set['NumberOfChildrenVisiting'])[0] # 0
Designation_out_index= outliers(train_set['Designation'])[0] # 89
MonthlyIncome_out_index= outliers(train_set['MonthlyIncome'])[0] # 138
'''
lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
                              
print(len(lead_outlier_index)) #577
lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)
'''       
x = train_set.drop(['ProdTaken',
                          'NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                        #   'TypeofContact',
                          ], axis=1)
# x = train_set.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                        #   'TypeofContact',
                          ], axis=1)
y = train_set['ProdTaken']
print(x.shape) #1911,13


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)

n_splits = 6
# 최상의 점수 :  0.9044520547945205
# acc : 0.954248366013072
# 걸린 시간 : 5.827547073364258 
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
# 2. 모델
###

tpot = TPOTClassifier(verbosity=2, random_state=72, cv=kfold, scoring='accuracy',warm_start=True, generations=10, population_size=100, early_stop=10)
tpot.fit(x_train, y_train)
####

print(tpot.score(x_test, y_test))
tpot.export('tpot_digits_pipeline.py')

import time 
y_predict = tpot.predict(x_test)
results = accuracy_score(y_test,y_predict)
print('acc :',results)

y_summit = tpot.predict(test_set)
y_summit = np.round(y_summit,0)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                      )
submission['ProdTaken'] = y_summit

submission.to_csv('TPOT3.csv',index=False)


##########
# 최상의 점수 :  0.8930338463986
# acc : 0.9418604651162791
# 걸린 시간 : 11.291642665863037

############ RandomState = 100
# 최상의 점수 :  0.8813139873889755
# acc : 0.921875


# 최적의 매개변수 :  {'od_pval': 0.12673190617341812, 'learning_rate': 0.01, 'l2_leaf_reg': 0.33021257848638497, 'fold_permutation_block': 142, 'depth': 8}
# 최상의 점수 :  0.9111839111839112
# acc : 0.9715909090909091
# 걸린 시간 : 6.49209451675415

# Best pipeline: ExtraTreesClassifier(XGBClassifier(MaxAbsScaler(input_matrix), learning_rate=0.01, max_depth=1, min_child_weight=5, n_estimators=100, n_jobs=1, subsample=0.45, verbosity=0), bootstrap=False, criterion=entropy, max_features=0.3, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
#   warnings.warn(
# 0.9715909090909091
# acc : 0.9715909090909091