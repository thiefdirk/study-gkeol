#컬럼 7개 이상 거래량 반드시 들어감 lstm 무적권 들어감
#삼성전자랑 앙상블


from colorsys import yiq_to_rgb
import xdrlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling1D, GRU, Activation, Dense, Conv1D, Reshape, LSTM, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
import time
start = time.time()


#1. 데이터
path = './_data/test_amore_0718/'
datasets1 = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949') 
datasets2 = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949') 


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



####################원핫인코더###################
# df1 = pd.DataFrame(y_train)
# df2 = pd.DataFrame(y_test)
# print(df1)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
df_요일1 = pd.DataFrame(oh.fit_transform(datasets1[['일자_요일']]), columns=['월', '화', '수', '목', '금'])
df_요일2 = pd.DataFrame(oh.transform(datasets2[['일자_요일']]), columns=['월', '화', '수', '목', '금'])
# df_요일1 = pd.concat([datasets1['일자'],df_요일1], axis=1)
# df_요일2 = pd.concat([datasets2['일자'],df_요일2], axis=1)
print(datasets1.shape)
print(datasets2.shape)
print(df_요일1.shape)
print(df_요일2.shape)


# df_요일1 = df_요일1.set_index('일자')
# df_요일2 = df_요일2.set_index('일자')

print(datasets1)

datasets1 = datasets1.drop(['일자_요일'], axis=1)
datasets2 = datasets2.drop(['일자_요일'], axis=1)

# print('====================================')
# print(datasets1.shape)
# print('====================================')
# print(datasets2.shape)
################################################

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
                            '일자_연', '일자_월', '일자_일'], axis=1)

datasets2d = np.array(datasets2d)

datasets2_50x = datasets2d[:1035,:]
datasets2_1x = datasets2d[1035:1771,:]
print(datasets2.shape)
print(datasets2_50x.shape)
print(datasets2_1x)
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


# print(datasets1.isnull().sum())
# print(datasets2.isnull().sum())

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scale_cols = ['시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비', '일자_연', '일자_월', '일자_일']
datasets1 = scaler.fit_transform(datasets1[scale_cols])
datasets2 = scaler.transform(datasets2[scale_cols])
# df_test = scaler.transform(df_test)



print(datasets1.shape, datasets2.shape)

datasets1 = pd.DataFrame(datasets1)
datasets2 = pd.DataFrame(datasets2)
print(datasets2)
datasets2.columns = ['시가', '고가', '저가', '종가', '전일비', '등락률',
                     '거래량', '금액(백만)',
                     '신용비','개인','기관',
                     '외인(수량)','외국계','프로그램','외인비',
                     '일자_연', '일자_월', '일자_일']
print(datasets2)
datasets2 = datasets2.drop(['신용비','개인','기관',
                            '외인(수량)','외국계','프로그램','외인비'], axis=1)

datasets1.columns = ['시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '신용비','개인','기관','외인(수량)','외국계',
                     '프로그램','외인비', '일자_연', '일자_월', '일자_일']

datasets2.columns = ['시가','고가','저가','종가',
                     '전일비','등락률','거래량','금액(백만)',
                     '일자_연', '일자_월', '일자_일']




datasets1 = pd.concat([datasets1,df_요일1], axis=1)
datasets2 = pd.concat([datasets2,df_요일2], axis=1)




datasets1 = datasets1.astype(dtype='float64')
datasets2 = datasets2.astype(dtype='float64')

print(index_1)
print(index_2)

datasets1 = datasets1.set_index([index_1])
datasets2 = datasets2.set_index([index_2])

print(datasets1)
print('====================================')
print(datasets2)
print('====================================')
print(datasets1.shape)
print('====================================')
print(datasets2.shape)

print(datasets1.iloc[1771])
print(datasets2.iloc[1035])

datasets1 = np.array(datasets1)
datasets2 = np.array(datasets2)

datasets1 = datasets1[:1771,:]
datasets2 = datasets2[:1771,:]
print(datasets1.shape, datasets2.shape) #(1771, 23) (1771, 23)
print(datasets1, datasets2)


print('====================================')
print(datasets2)
x1 = np.transpose(datasets1) 
x2 = np.transpose(datasets2)
print(datasets2)
print(datasets1)



x1 = np.transpose(datasets1) 
x2 = np.transpose(datasets2)
x = datasets1[:,:-1]
y = datasets1[:,-1]

print(x1.shape, x2.shape) # (23, 1771) (16, 1771)

###################리세이프#######################
x1 = x1.reshape(22, 77, 23)
x2 = x2.reshape(16, 77, 23)
# df_test = df_test.reshape(14, 647, 650)
# print(np.unique(y_train, return_counts=True))
#################################################

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(x1_train.shape, x1_test.shape) # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape) # (80, 3) (20, 3)
print(y1_train.shape, y1_test.shape) # (80,) (20,)
print(y2_train.shape, y2_test.shape) # (80,) (20,)


#2. 모델

#2-1. 모델1
input1 = Input(shape=(77,23))
dense1 = Dense(100, activation='relu', name='ak1')(input1)
dense2 = Dense(200, activation='relu', name='ak2')(dense1)
dense3 = Dense(300, activation='relu', name='ak3')(dense2)
output1 = Dense(100, activation='relu', name='out_ak1')(dense3)

#2-2. 모델2
input2 = Input(shape=(77,23))
dense11 = Dense(1100, activation='relu', name='ak11')(input2)
dense12 = Dense(120, activation='relu', name='ak12')(dense11)
dense13 = Dense(130, activation='relu', name='ak13')(dense12)
dense14 = Dense(140, activation='relu', name='ak14')(dense13)
output2 = Dense(100, activation='relu', name='out_ak2')(dense14)

from tensorflow.python.keras.layers import concatenate, Concatenate # 앙상블모델
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(20, activation='relu', name='mg2')(merge1)
merge3 = Dense(300, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

print(merge1.shape)

model = Model(inputs=[input1, input2], outputs=last_output)
