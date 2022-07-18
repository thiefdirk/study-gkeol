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
datasets1 = pd.read_csv(path + '아모레220718.csv', encoding='cp949') 
datasets2 = pd.read_csv(path + '삼성전자220718.csv', encoding='cp949') 



datasets1 = pd.DataFrame(datasets1)
datasets2 = pd.DataFrame(datasets2)

datasets1 = datasets1.drop(datasets1.columns[[5]], axis=1)
datasets2 = datasets2.drop(datasets2.columns[[5]], axis=1)

datasets1 = datasets1.dropna(axis=0)
datasets2 = datasets2.dropna(axis=0)
print(datasets1)


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

datasets1 = datasets1.set_index('일자')
datasets2 = datasets2.set_index('일자')

datasets1 = datasets1.drop(['일자_datetime'], axis=1)
datasets2 = datasets2.drop(['일자_datetime'], axis=1)

print(datasets1.describe())
print(datasets1.info())

print(datasets1)
datasets1 = datasets1.astype('float64')
datasets2 = datasets2.astype('float64')

x1 = np.transpose(datasets1) 
x2 = np.transpose(datasets2)

print(x1.shape, x2.shape) # (16, 3170) (16, 3037)

####################원핫인코더###################
# df1 = pd.DataFrame(y_train)
# df2 = pd.DataFrame(y_test)
# print(df1)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
datasets1 = pd.DataFrame(oh.fit_transform(datasets1[['일자_요일']]), columns=['월', '화', '수', '목', '금'])
datasets2 = pd.DataFrame(oh.transform(datasets2[['일자_요일']]), columns=['월', '화', '수', '목', '금'])
print('====================================')
print(datasets1.shape)
print('====================================')
print(datasets2.shape)
################################################

# print(datasets1.isnull().sum())
# print(datasets2.isnull().sum())

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)
x2 = scaler.transform(x2)
# df_test = scaler.transform(df_test)

# x = df[:,:-1]
# y = df[:,-1]
