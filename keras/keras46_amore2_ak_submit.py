#컬럼 7개 이상 거래량 반드시 들어감 lstm 무적권 들어감
#삼성전자랑 앙상블
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################

from colorsys import yiq_to_rgb
import xdrlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import MaxPooling1D, GRU, Activation, Dense, Conv1D, Reshape, LSTM, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
import time
start = time.time()


#1. 데이터
path = './_data/test_amore_0718/'
datasets1 = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949') 
datasets2 = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949') 
datasets1 = datasets1.sort_values(by='일자', ascending=True) #오름차순 정렬
datasets2 = datasets2.sort_values(by='일자', ascending=True) #오름차순 정렬

datasets1 = datasets1.drop(datasets1.columns[[5]], axis=1)
datasets2 = datasets2.drop(datasets2.columns[[5]], axis=1)

datasets1 = datasets1.dropna(axis=0)
datasets2 = datasets2.dropna(axis=0)



datasets1.columns = ['일자','시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비']
datasets2.columns = ['일자','시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비']
print(datasets1)

pd.to_datetime(datasets1['일자'])
pd.to_datetime(datasets2['일자'])
print(datasets1)



datasets1['일자_datetime'] = pd.to_datetime(datasets1['일자'])
datasets2['일자_datetime'] = pd.to_datetime(datasets2['일자'])


datasets1['일자_연'] = datasets1['일자_datetime'].dt.year
datasets1['일자_월'] = datasets1['일자_datetime'].dt.month
datasets1['일자_일'] = datasets1['일자_datetime'].dt.day
datasets1['일자_요일'] = datasets1['일자_datetime'].dt.day_name()

datasets2['일자_연'] = datasets2['일자_datetime'].dt.year
datasets2['일자_월'] = datasets2['일자_datetime'].dt.month
datasets2['일자_일'] = datasets2['일자_datetime'].dt.day
datasets2['일자_요일'] = datasets2['일자_datetime'].dt.day_name()



datasets1 = datasets1.drop(['일자_datetime'], axis=1)
datasets2 = datasets2.drop(['일자_datetime'], axis=1)

print(datasets1.describe())
print(datasets1.info())

print(datasets1)

day_name1 = datasets1[['일자_요일']]

day_name2 = datasets2[['일자_요일']]
print(day_name1)
print(day_name2)
day_name1 = np.array(day_name1)
day_name2 = np.array(day_name2)

day_name2 = day_name2[:1771,:]


index_1 = np.array(datasets1['일자'])
index_2 = np.array(datasets2['일자'])


print(datasets1)
print('====================================')
print(datasets2)
print('====================================')
print(datasets1.shape)
print('====================================')
print(datasets2.shape)


print(datasets1.columns, datasets2.columns)

drop_columns = datasets2[['일자','신용비','개인','기관',
                            '외인(수량)','외국계','프로그램','외인비',
                            '일자_연', '일자_월', '일자_일']]
drop_columns = np.array(drop_columns)
drop_columns = drop_columns[:1771,:]

print(datasets2.shape)
datasets2d = datasets2.drop(['일자','신용비','개인','기관',
                            '외인(수량)','외국계','프로그램','외인비',
                            '일자_연', '일자_월', '일자_일', '일자_요일'], axis=1)

datasets2d = np.array(datasets2d)

datasets2_50x = datasets2d[-1035:,:]
datasets2_1x = datasets2d[-1771:-1035,:]
print(datasets2.shape)
print(datasets2_50x.shape) #(1035, 8)
print(datasets2_1x.shape) #(736, 8)
datasets2_1x = datasets2_1x*(50)
print(datasets2_1x)
datasets2_50x = pd.DataFrame(datasets2_50x)
datasets2_1x = pd.DataFrame(datasets2_1x)
drop_columns = pd.DataFrame(drop_columns)
print(datasets2_50x)
print(datasets2_1x)

datasets2dd = pd.concat([datasets2_50x,datasets2_1x], axis=0)
print(datasets2dd)
print(drop_columns)
datasets2dd = datasets2dd.reset_index()
drop_columns = drop_columns.reset_index()
datasets2 = pd.concat([datasets2dd, drop_columns], axis=1,ignore_index=False)
datasets2 = datasets2.drop(['index'], axis=1)

print(datasets2dd.shape)
print(drop_columns.shape)
print(datasets2)
datasets2.columns = ['시가', '고가', '저가', '종가', '전일비', '등락률',
                     '거래량', '금액(백만)',
                     '일자','신용비','개인','기관',
                     '외인(수량)','외국계','프로그램','외인비',
                     '일자_연', '일자_월', '일자_일']
print(datasets2)
print(datasets1.columns)
# Index(['일자', '시가', '고가', '저가', '종가', '전일비', '등락률', '거래량', '금액(백만)', '신용비',
#        '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비', '일자_연', '일자_월', '일자_일',
#        '일자_요일'],
#       dtype='object')
y1 = datasets1[['종가']]
y2 = datasets2[['종가']]

# print(datasets1.isnull().sum())
# print(datasets2.isnull().sum())

scaler1 = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scale_cols = ['시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비', '일자_연', '일자_월', '일자_일']
datasets1 = scaler1.fit_transform(datasets1[scale_cols])
# df_test = scaler.transform(df_test)

scaler2 = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scale_cols = ['시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비', '일자_연', '일자_월', '일자_일']
datasets2 = scaler2.fit_transform(datasets2[scale_cols])



print(datasets1.shape, datasets2.shape)

datasets1 = pd.DataFrame(datasets1)
datasets2 = pd.DataFrame(datasets2)
day_name1 = pd.DataFrame(day_name1)
day_name2 = pd.DataFrame(day_name2)
print(datasets1)
print(datasets2)
print(day_name1)
print(day_name2)
datasets1 = datasets1.reset_index()
datasets2 = datasets2.reset_index()
day_name1 = day_name1.reset_index()
day_name2 = day_name2.reset_index()
print(datasets1)
print(datasets2)
print(day_name1)
print(day_name2)
datasets1 = datasets1.drop(['index'], axis=1)
datasets2 = datasets2.drop(['index'], axis=1)
day_name1 = day_name1.drop(['index'], axis=1)
day_name2 = day_name2.drop(['index'], axis=1)
print(datasets1)
print(datasets2)
print(day_name1)
print(day_name2)
datasets1 = pd.concat([datasets1, day_name1], axis=1,ignore_index=True)
datasets2 = pd.concat([datasets2, day_name2], axis=1,ignore_index=True)
print(datasets1)
print(datasets2)





datasets2.columns = ['시가', '고가', '저가', '종가', '전일비', '등락률',
                     '거래량', '금액(백만)',
                     '신용비','개인','기관',
                     '외인(수량)','외국계','프로그램','외인비',
                     '일자_연', '일자_월', '일자_일','일자_요일']
print(datasets2)
datasets2 = datasets2.drop(['신용비','개인','기관',
                            '외인(수량)','외국계','프로그램','외인비'], axis=1)

####################라벨인코더###################
encoder = LabelEncoder()
# df1 = pd.DataFrame(y_train)
# df2 = pd.DataFrame(y_test)
# print(df1)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
df_요일1 = pd.DataFrame(encoder.fit_transform(day_name1))
df_요일2 = pd.DataFrame(encoder.transform(day_name2))
# df_요일1 = pd.concat([datasets1['일자'],df_요일1], axis=1)
# df_요일2 = pd.concat([datasets2['일자'],df_요일2], axis=1)
print(datasets1.shape)
print(datasets2.shape)
print(df_요일1.shape)
print(df_요일2.shape)


# df_요일1 = df_요일1.set_index('일자')
# df_요일2 = df_요일2.set_index('일자')

print(datasets1)

datasets1 = datasets1.drop([datasets1.columns[18]], axis=1)
datasets1 = datasets1.drop([datasets1.columns[3]], axis=1)
datasets2 = datasets2.drop(['일자_요일', '종가','일자_연','일자_월','일자_일'], axis=1)
print(datasets1)
print(datasets2)
# print('====================================')
# print(datasets1.shape)
# print('====================================')
# print(datasets2.shape)
################################################

datasets1.columns = ['시가','고가','저가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비', '일자_연', '일자_월', '일자_일']

datasets2.columns = ['시가','고가','저가',
                     '전일비','등락률','거래량','금액(백만)']

datasets1 = datasets1.drop(['신용비','개인','기관','외인(수량)','외국계','프로그램','외인비'], axis=1)


print(datasets1)
print(datasets2)

datasets1 = pd.concat([datasets1,df_요일1], axis=1)
# datasets2 = pd.concat([datasets2,df_요일2], axis=1)
print('====================================')

print(datasets1)
print(datasets2)

print('====================================')

datasets1 = datasets1.astype(dtype='float64')
datasets2 = datasets2.astype(dtype='float64')

print(index_1)
print(index_2)

# datasets1 = datasets1.set_index([index_1])
# datasets2 = datasets2.set_index([index_2])

print(datasets1)
print('====================================')
print(datasets2)
print('====================================')
print(datasets1.shape)
print('====================================')
print(datasets2.shape)
print(datasets1.columns, datasets2.columns)
print(datasets1.iloc[1771])
print(datasets2.iloc[1035])

datasets1 = np.array(datasets1)
datasets2 = np.array(datasets2)

datasets1 = datasets1[-1771:,:]
datasets2 = datasets2[-1771:,:]
print(datasets1.shape, datasets2.shape) #(1771, 22) (1771, 16)
print(datasets1, datasets2)

# predict_set1 = datasets1[-301:-1,:]
# predict_set2 = datasets2[-301:-1,:]

# summit_set1 = datasets1[-300:,:]
# summit_set2 = datasets2[-300:,:]
# print(predict_set1.shape, predict_set2.shape) #(300, 22) (300, 16)
# print(summit_set1.shape, summit_set2.shape) # (300, 22) (300, 16)
print('====================================')

print(datasets1)
print(datasets2)
# print(predict_set1)
# print(predict_set2)
# print(summit_set1)
# print(summit_set2)

datasets1 = pd.DataFrame(datasets1)
datasets2 = pd.DataFrame(datasets2)
y1 = np.array(y1)
y1 = y1[-1771:,:]
y1 = pd.DataFrame(y1)

print(y2.shape) #(1771, 1)
y2 = np.array(y2)
y2 = y2[-1771:,:]
y2 = pd.DataFrame(y2)

datasets1 = pd.concat([datasets1,y1], axis=1)
datasets1 = np.array(datasets1)

datasets2 = pd.concat([datasets2,y2], axis=1)
datasets2 = np.array(datasets2)

print(datasets1)
print(datasets2)

def split_xy3(dataset, time_step, y_columns):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_step
        y_end_number = x_end_number + y_columns - 1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy3(datasets1, 20, 3)
print(x1)
print(x1.shape) # (1752, 20, 18)


x2, y2 = split_xy3(datasets2, 20, 3)
print(x2)
print(x2.shape) # (1752, 20, 12)

print(x1.shape, x2.shape) # (1750, 20, 18) (1750, 20, 11)
print(y1.shape, y2.shape) # (1771, 1) (1750, 3)







# predict_set1 = split_x(predict_set1, size)
# predict_set2 = split_x(predict_set2, size)
# summit_set1 = split_x(summit_set1, size)
# summit_set2 = split_x(summit_set2, size)
# print(y.shape) # (1752, 20, 1)
# print(predict_set1.shape) # (281, 20, 22)
# print(predict_set2.shape) # (281, 20, 16)
# print(summit_set1.shape) # (281, 20, 16)
# print(summit_set2.shape) # (281, 20, 16)
# print(summit_set2.shape) # (281, 20, 16)
# print(y2.shape) # (281, 20, 1)


# ###################리세이프#######################
# x1 = x1.reshape(1752, 20, 22)
# x2 = x2.reshape(1752, 20, 16)
# test_set1 = test_set1.reshape(281, 20, 22)
# test_set1 = test_set1.reshape(281, 20, 22)
# y = y.reshape(1752, 20, 1)
# # print(np.unique(y_train, return_counts=True))
# #################################################

x1_train, x1_test, x2_train, x2_test, \
y1_train, y1_test = train_test_split(x1,x2,y1,train_size=0.7,
                                                    shuffle=False
                                                    )

print(x1_train.shape, x1_test.shape) # (1225, 20, 11) (525, 20, 11)
print(x2_train.shape, x2_test.shape) # (1225, 20, 7) (525, 20, 7)
print(y1_train.shape, y1_test.shape) # (1225, 3) (525, 3)
# print(y2_train.shape, y2_test.shape) # (1225, 3) (525, 3)


# #2. 모델

# #2-1. 모델1
# input1 = Input(shape=(20,18))
# conv1D_0 = Conv1D(400, 2, activation='relu')(input1)
# conv1D_1 = Bidirectional(LSTM(100, return_sequences=True))(conv1D_0)
# conv1D_2 = Bidirectional(LSTM(80))(conv1D_1)
# dense1 = Dense(20)(conv1D_2)
# batchnorm1 = BatchNormalization()(dense1)
# activ1 = Activation('relu')(batchnorm1)
# dense2 = Dense(100, activation='relu')(activ1)
# output1 = Dense(50)(dense2)
# model = Model(inputs=input1, outputs=output1)   

# #2-2. 모델2
# input2 = Input(shape=(20,12))
# conv1D_10 = Conv1D(400, 2, activation='relu')(input2)
# conv1D_11 = Bidirectional(LSTM(100))(conv1D_10)
# dense11 = Dense(20)(conv1D_11)
# batchnorm11 = BatchNormalization()(dense11)
# activ11 = Activation('relu')(batchnorm11)
# dense12 = Dense(100, activation='relu')(activ11)
# output2 = Dense(30)(dense12)
# model = Model(inputs=input2, outputs=output2)   

# from tensorflow.python.keras.layers import concatenate, Concatenate # 앙상블모델
# merge1 = concatenate([output1, output2], name='mg1')
# merge2 = Dense(120, activation='relu', name='mg2')(merge1)
# merge3 = Dense(30, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)

# print(merge1.shape)

# model = Model(inputs=[input1, input2], outputs=last_output)


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M") # 0707_1723
# print(date)

# save_filepath = './_ModelCheckPoint/' + current_name + '/'
# load_filepath = './_ModelCheckPoint/' + current_name + '/'

model = load_model('./_ModelCheckPoint/keras46_amore2_ak_save.py/0719_1940_0011-1897107456.0000.hdf5')


# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
#                               restore_best_weights=True)        

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([save_filepath, date, '_', filename])
#                       )

# hist = model.fit([x1_train,x2_train] ,y_train, epochs=1000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping,mcp],
#                  verbose=1)

#4. 평가, 예측

loss = model.evaluate([x1_test,x2_test],y1_test)
y_predict2 = model.predict([x1_test,x2_test])
# y_summit = model.predict([summit_set1,summit_set2])
# y_predict=np.array(y_predict)
# y_predict = np.array(y_predict) #(2, 20, 1)
# print(y_predict) 
# print(y_test.shape)
# print(np.array(y_test))
# y_test=np.array(y_test)
# print(y_predict)
# print(y)
# print(y.shape)
# print(y_test.shape)
# print(y_predict.shape)
# y_test = y_test.reshape(526, 20)

# r2 = r2_score(y_test, y_predict)
print('loss: ', loss)
# print('r2스코어 : ', r2)
print('내일 종가 : ', y_predict2[-1:])
print("time :", time.time() - start)

# loss:  [1542844544.0, 31428.939453125]
# 내일 종가 :  [[135789.28]]
# time : 114.78367066383362

# loss:  [7308519424.0, 66982.984375]
# 내일 종가 :  [[131833.27]]
# time : 7.877325057983398