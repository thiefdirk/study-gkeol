import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩


import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

print(y_test)
print(y_train)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 로지스틱분류, 분류
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier # 의사결정트리
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트

model = LinearSVC() 
# model = LogisticRegression() # 로지스틱분류
# model = KNeighborsClassifier() 
# model = DecisionTreeClassifier() 
# model = RandomForestClassifier() 

#3. 컴파일 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 acc : ', result)

y_predict = model.predict(x_test)
acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', label='loss', color='red')
# plt.plot(hist.history['val_loss'], marker='.', label='val_loss', color='blue')
# plt.grid()
# plt.title('시그모이드')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()

# loss :  0.05714249983429909
# accuracy :  1.0

# 
# LogisticRegression 모델의 정확도 :  1.0
# KNN 모델의 정확도 :  0.9666666666666667
# DecisionTreeClassifier 모델의 정확도 :  0.9333333333333333
# RandomForestClassifier 모델의 정확도 :  0.9333333333333333