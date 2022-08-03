from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
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
from sklearn.datasets import fetch_covtype
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]


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
y_predict = np.argmax(y_predict, axis= 1)  # 판다스 겟더미 쓸때는 tf.argmax sklearn 원핫인코딩 쓸때는 np
y_test = np.argmax(y_test, axis= 1)
# y_predict = to_categorical(y_predict)
# y_test = np.argmax(y_test, axis= 1)


acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

# accuracy :  0.7124047640903249

# 스케일러 하기전 acc스코어 :  0.781129520837158  loss :  0.5174677968025208
# 민맥스 acc스코어 :  0.8581902882320543  loss :  0.3478740453720093
# 스탠다드 acc스코어 :  0.8729346429227097  loss :  0.31877365708351135

# maxabs acc스코어 :  0.8425222599596108    loss :  0.3773805499076843
# robust acc스코어 :  0.869274371213512    loss :  0.3183009922504425