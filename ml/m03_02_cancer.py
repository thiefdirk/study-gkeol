from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
import time

#1. 데이터
datasets = load_breast_cancer()
# print(datasets) (569,30)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data   #['data']
y = datasets.target #['target']
print(x.shape, y.shape) # (569,30), (569,)


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


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 로지스틱분류, 분류
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.tree import DecisionTreeClassifier # 의사결정트리
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트

# model = LinearSVC() 
# model = LogisticRegression() # 로지스틱분류
# model = KNeighborsClassifier() 
# model = DecisionTreeClassifier() 
model = RandomForestClassifier() 
                  
                           
#3. 컴파일, 훈련

hist = model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('acc : ', result)
print('결과 : ', y_predict)

# LinearSVC : 0.9766081871345029
# LogisticRegression : 0.9883040935672515
# KNeighborsClassifier : 0.9590643274853801
# DecisionTreeClassifier : 0.9590643274853801
# RandomForestClassifier : 0.9649122807017544