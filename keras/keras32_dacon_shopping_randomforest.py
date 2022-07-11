from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################


#1. 데이터
path = './_data/dacon_shopping/'
train_set = pd.read_csv(path + 'train.csv' # + 명령어는 문자를 앞문자와 더해줌
                        ) # index_col=n n번째 컬럼을 인덱스로 인식
Weekly_Sales = train_set[['Weekly_Sales']]
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv' # 예측에서 쓸거임                
                       )
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.columns)
print(train_set.info()) # info 정보출력
print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum().sort_values(ascending=False)



######## 년, 월 ,일 분리 ############

train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.Date)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.Date)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.Date)]

test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.Date)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.Date)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.Date)]

train_set.drop(['id', 'Date','Weekly_Sales'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍
test_set.drop(['id', 'Date'],axis=1,inplace=True) # 트레인 세트에서 데이트타임 드랍

print(train_set)
print(test_set)

##########################################

# ####################원핫인코더###################

df = pd.concat([train_set, test_set])
print(df)

print(df)

alldata = pd.get_dummies(df, columns=['day','Store','month', 'year'])
print(alldata)

train_set2 = alldata[:len(train_set)]
test_set2 = alldata[len(train_set):]

print(train_set2)
print(test_set2)
# train_set = pd.get_dummies(train_set, columns=['Store','month', 'year', 'IsHoliday'])
# test_set = pd.get_dummies(test_set, columns=['Store','month', 'year', 'IsHoliday'])




###############프로모션 결측치 처리###############

train_set2 = train_set2.fillna(0)
test_set2 = test_set2.fillna(0)

print(train_set2)
print(test_set2)

##########################################

train_set2 = pd.concat([train_set2, Weekly_Sales],axis=1)
print(train_set2)

x = train_set2.drop(['Weekly_Sales'], axis=1)
y = train_set2['Weekly_Sales']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )


# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_set2 = scaler.transform(test_set2)

# # print(test_set2)
# print(x_train.info())
# print(y_train.info())
# print(train_set2.info())

x_train = x_train.values
y_train = y_train.values

x_train.astype('int')
y_train.astype('int')

print(x_train.shape)
print(y_train.shape)


# 2. 모델구성


model = RandomForestRegressor()
model.fit(x_train, y_train)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


'''
#3. 컴파일, 훈련


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=3000, batch_size=128,
                 validation_split=0.3,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)


# model = load_model(load_filepath + '0711_1732_2300-8791202816.발리데이션0.3.hdf5')
'''
#4. 평가, 예측

print("=============================1. 기본 출력=================================")
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)

print(test_set2)

y_summit = model.predict(test_set2)

print(y_summit)
print(y_summit.shape) # (180, 1)

submission_set = pd.read_csv(path + 'sample_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
                             index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

print(submission_set)

submission_set['Weekly_Sales'] = y_summit
print(submission_set)


submission_set.to_csv(path + 'submission_randomforest.csv', index = True)

# 민맥스
# loss :  [12031329280.0, 60506.640625]
# RMSE :  109687.42480795768
# r2스코어 :  0.9654250161340566

# 스탠다드
# loss :  [13667978240.0, 65182.3046875]
# RMSE :  116910.13121703712
# r2스코어 :  0.9607217073891772

# 로버스트
# loss :  [19251437568.0, 76360.9453125]
# RMSE :  138749.56208955363
# r2스코어 :  0.9446762579822676

# 맥스앱스
# loss :  [14230065152.0, 60868.4140625]
# RMSE :  119289.83618940222
# r2스코어 :  0.9591064145913685

# 트레인사이즈 0.7 발리데이션 0.3 민맥스
# loss :  [11345984512.0, 60342.859375]
# RMSE :  106517.52817997398
# r2스코어 :  0.966259471816233

# 트레인사이즈 0.8 발리데이션 0.2 민맥스 epochs 2500 페이션스 300
# loss :  [13296029696.0, 59877.20703125]
# RMSE :  115308.41749125942
# r2스코어 :  0.9617905902393659

# 트레인사이즈 0.8 발리데이션 0.1 민맥스 epochs 2500 페이션스 300
# loss :  [14210052096.0, 60650.4921875]
# RMSE :  119205.92252226875
# r2스코어 :  0.9591639270145448


# loss :  [11314745344.0, 59876.4375]
# RMSE :  106370.79280402962
# r2스코어 :  0.9663523676812774

# 랜덤포레스트