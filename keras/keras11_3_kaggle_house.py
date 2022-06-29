import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from keras.layers.recurrent import LSTM, SimpleRNN
import datetime as dt

encording_columns = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',
                    'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
                    'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
                    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
                    'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

non_encording_columns = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond',
                         'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
                         'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
                         'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                         'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea',
                         'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                         'MiscVal','MoSold','YrSold']



#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv', index_col=0) # 예측에서 쓸거임  3

# 수치형 변수와 범주형 변수 찾기
numerical_feats = train_set.dtypes[train_set.dtypes != "object"].index
categorical_feats = train_set.dtypes[train_set.dtypes == "object"].index
numerical_feats_ = test_set.dtypes[test_set.dtypes != "object"].index
categorical_feats_ = test_set.dtypes[test_set.dtypes == "object"].index
# print("Number of Numberical features: ", len(numerical_feats)) # 37
# print("Number of Categorical features: ", len(categorical_feats)) # 43
'''
# 변수명 출력
print(train_set[numerical_feats].columns)      
print("*"*79)
print(train_set[categorical_feats].columns)      
print(test_set[numerical_feats_].columns)      
print("*"*79)
print(test_set[categorical_feats_].columns)   
'''
train_set_encoded = train_set.drop(numerical_feats,axis=1)
print(train_set_encoded)

test_set_encoded = test_set.drop(numerical_feats_,axis=1)
print(test_set_encoded)

le = LabelEncoder()

train_set_encoded.loc[:,:] = \
train_set_encoded.loc[:,:].apply(LabelEncoder().fit_transform)    

print(train_set_encoded)

train_set = pd.concat([train_set_encoded, train_set.loc[:,numerical_feats]], axis=1)

print(train_set)

'''


le = le.fit(train_set[categorical_feats])   #train['col']을 fit
train_set = le.transform(train_set[categorical_feats])
print(train_set)





# One-hot Encoding
ohe = OneHotEncoder(sparse=False)
ohe.fit(categorical_feats)

# 인코딩한 데이터로 변환
ohe_encoded = ohe.transform(categorical_feats)
new_cat = pd.DataFrame(ohe_encoded)
categorical_feats = pd.concat([categorical_feats, new_cat], axis=1)

ohe_encoded_ = ohe.transform(categorical_feats_)
new_cat_ = pd.DateFrame(ohe_encoded_)
categorical_feats_ = pd.concat([categorical_feats_, new_cat_], axis=1)

x = train_set.drop(['saleprice'], axis=1)
y = train_set['saleprice']

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )

print(x_train)
print(x_test)
print(test_set)


print(train_set)
print(train_set.shape) # (1460, 80)
print(train_set.columns)
print(train_set.info()) # info 정보출력 non-null
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력     
           
print(test_set)
print(test_set.shape) # (1459, 79)
print(test_set.info()) # non-null

x = train_set.drop(['SalePrice'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1460, 79)

y = train_set['SalePrice'] 
print(y)
print(y.shape) # (1460,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=31
                                                    )

print(x_train)
print(x_test)
print(test_set)



le.fit(x_train[encording_columns])   #train['col']을 fit
x_train[encording_columns] = le.transform(x_train[encording_columns])   #train['col']에 따라 encoding
x_test[encording_columns] = le.transform(x_train[encording_columns])   #train['col']에 따라 encoding                    
test_set[encording_columns] = le.transform(x_train[encording_columns])   #train['col']에 따라 encoding
                    
print(x_train)
print(x_test)
print(test_set)
'''