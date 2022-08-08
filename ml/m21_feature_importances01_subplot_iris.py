import numpy as np
from sklearn.datasets import load_iris, load_diabetes

# 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#2. 모델구성 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

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

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4.평가예측
# result = model.score(x_test, y_test)
# print("model.score:",result)

# from sklearn.metrics import accuracy_score, r2_score
# y_predict =model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print('r2_score:', r2)

print("===================================")
print(model1,':',model1.feature_importances_)
print(model2,':',model2.feature_importances_)
print(model3,':',model3.feature_importances_)
print(model4,':',model4.feature_importances_)

import matplotlib.pyplot as plt

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
        plt.title('XGB()')
    else :
        plt.title(model_list[i])

plt.show()

# DecisionTreeClassifier() : 
# model.score: -0.057892604418324556
# accuracy_score: -0.057892604418324556
# [0.08029344 0.01446961 0.34390074 0.07699374 0.03310105 0.09085236 
# 0.05462762 0.01573113 0.16389064 0.12613968]

# GradientBoostingClassifier() : 
# model.score: 0.4241422041142602
# accuracy_score: 0.4241422041142602
# [0.05926608 0.01410165 0.32622491 0.08454118 0.0466377  0.06082818
#  0.06078631 0.02708192 0.23918247 0.08134961]

# RandomForestClassifier() : 
# model.score: 0.41928164399730516
# accuracy_score: 0.41928164399730516
# [0.04591924 0.01652454 0.33587663 0.09600231 0.03143398 0.06647812
#  0.0383661  0.01419611 0.27648568 0.07871729]

#  XGBClassifier [0.00912187 0.0219429  0.678874   0.29006115]
# model.score: 0.2602960708365062
# accuracy_score: 0.2602960708365062
# [0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191
#  0.06551369 0.17944618 0.13779876 0.08540721]
