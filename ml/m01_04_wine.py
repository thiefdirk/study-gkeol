from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_wine
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13), (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델
model = LinearSVC()

#3. 컴파일 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

results= model.score(x_test, y_test)
print('accuracy : ', results)
y_predict = model.predict(x_test)


# 스케일러 하기전 acc스코어 :  0.9444444444444444  loss :  0.2627679407596588
# 민맥스 acc스코어 :  0.9814814814814815  loss :  0.02856956608593464
# 스탠다드 acc스코어 :  0.9629629629629629  loss :  0.07545127719640732

# maxabs acc스코어 :  0.9629629629629629    loss :  0.10922898352146149
# robust acc스코어 :  0.9629629629629629    loss :  0.06978224962949753

# accuracy :  0.9814814814814815