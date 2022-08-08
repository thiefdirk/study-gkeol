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



# 결과비교
# 1. DecisionTree
# 기존 acc
# 컬럼삭제후 acc
import numpy as np
from sklearn.datasets import load_iris, load_diabetes

# 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# 컬럼명 확인
print(datasets.feature_names)
# 컬럼 제거

# x = x[:, [2, 3]] # 첫번째, 두번째 컬럼 제거


from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#2. 모델구성 

# model1 = DecisionTreeRegressor()
# model2 = RandomForestRegressor()
# model3 = GradientBoostingRegressor()
# model4 = XGBRegressor()

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

model_list = [model1, model2, model3, model4]

#3. 훈련

# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
# model3.fit(x_train,y_train)
# model4.fit(x_train,y_train)

#3. 훈련, 평가예측

for i in range(len(model_list)) :
    model_list[i].fit(x_train,y_train)
    result = model_list[i].score(x_test, y_test)
    print("model.score:",result)
    y_predict =model_list[i].predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    print('accuracy_score:', acc)

print("===================================")
print(model1,':',model1.feature_importances_)
print(model2,':',model2.feature_importances_)
print(model3,':',model3.feature_importances_)
print(model4,':',model4.feature_importances_)


def plot_feature_importances(model) : 
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수직바
    plt.yticks(np.arange(n_features), datasets.feature_names) # 수직바 위쪽에 컬럼명 추가, arange는 수열을 만들어줌
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(model)


plt.figure(figsize=(8,8))
for i in range(len(model_list)):
    plt.subplot(2,2,i+1) # 2행 2열에서 i+1번째 칸에 그림을 그림
    plot_feature_importances(model_list[i])
    if str(model_list[i]).startswith('XGBClassifier'):
        plt.title('XGBClassifier()')
    else :
        plt.title(model_list[i])

for model in model_list:
    model_drop_cal = []
    model.fit(x_train, y_train)
    score_bf = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score_bf)
    else:
        print(str(model).strip('()'), '의 스코어: ', score_bf)
    for i in range(len(model.feature_importances_)):
        if model.feature_importances_[i]<=0.03:
            model_drop_cal.append(i)
        np.delete(x, model_drop_cal, axis=1)
    model.fit(x_train, y_train)
    score_af = model.score(x_test, y_test)    
    # print('중요도낮은칼럼: ', model_drop_cal)
    # print('모든칼럼중요도: ', model.feature_importances_)
    print('중요도낮은칼럼제외후점수: ', score_bf)
    print('중요도낮은칼럼제외후점수: ', score_af)
    


plt.show()


# DecisionTreeClassifier() : 1.0
# RandomForestClassifier() : 1.0
# GradientBoostingClassifier() : 1.0
# XGBClassifier() : 1.0
########################################컬럼 삭제 전, 후##############################################
# DecisionTreeClassifier() : 0.9666666666666667
# RandomForestClassifier() : 0.9666666666666667
# GradientBoostingClassifier() : 0.9666666666666667
# XGBClassifier() : 0.9666666666666667