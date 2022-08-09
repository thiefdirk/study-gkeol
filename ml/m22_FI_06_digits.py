# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후 
# 각 모델별로 돌려서 결과 도출!

# 기존모델결과와 비교

from os import access
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline # pipeline을 사용하기 위한 함수
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# 결과비교
# 1. DecisionTree
# 기존 acc
# 컬럼삭제후 acc
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, fetch_covtype, load_digits

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
print(np.unique(y)) # [1 2 3 4 5 6 7]

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))
# x = x[:, [2, 3]] # 첫번째, 두번째 컬럼 제거

#2. 모델구성
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

model_list = [model1, model2, model3, model4]

def plot_feature_importances(model) : 
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수직바
    plt.yticks(np.arange(n_features), datasets.feature_names) # 수직바 위쪽에 컬럼명 추가, arange는 수열을 만들어줌
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model)
print(np.unique(y)) # [1 2 3 4 5 6 7]

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#3. 훈련, 컴파일
for model in model_list:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)
    
    
plt.figure(figsize=(8,8))
for i in range(len(model_list)):
    plt.subplot(2,2,i+1) # 2행 2열에서 i+1번째 칸에 그림을 그림
    plot_feature_importances(model_list[i])
    if str(model_list[i]).startswith('XGBClassifier'):
        plt.title('XGBClassifier()')
    else :
        plt.title(model_list[i])
plt.show()

    

# DecisionTreeClassifier 의 스코어:  0.8694444444444445
# DecisionTreeClassifier 의 드랍후 스코어:  0.8583333333333333
# RandomForestClassifier 의 스코어:  0.9777777777777777
# RandomForestClassifier 의 드랍후 스코어:  0.9777777777777777
# GradientBoostingClassifier 의 스코어:  0.975
# GradientBoostingClassifier 의 드랍후 스코어:  0.975
# XGB 의 스코어:  0.975
# XGB 의 드랍후 스코어:  0.975