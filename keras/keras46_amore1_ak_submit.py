#컬럼 7개 이상 거래량 반드시 들어감 lstm 무적권 들어감
#삼성전자랑 앙상블
# -*- coding: utf-8 -*-
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
path = './_data/0718시험/'
datasets1 = pd.read_csv(path + 'amore220718.csv') 
datasets2 = pd.read_csv(path + 'samsung220718.csv') 

