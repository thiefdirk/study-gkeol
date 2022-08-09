import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes

# 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x = pd.DataFrame(x, columns=datasets.feature_names)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=True, random_state=1234)


#2. 모델구성 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

#3. 훈련

model.fit(x_train,y_train)

#4.평가예측
result = model.score(x_test, y_test)
print("model.score:",result)

from sklearn.metrics import accuracy_score, r2_score
y_predict =model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score:', r2)

print("===================================")
print(model,':',model.feature_importances_)

import matplotlib.pyplot as plt

# def plot_feature_importances(model) : 
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center') # 수직바
#     plt.yticks(np.arange(n_features), datasets.feature_names) # 수직바 위쪽에 컬럼명 추가, arange는 수열을 만들어줌
#     plt.xlabel("Feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)
#     plt.title(model)
#     plt.show()
    
# plot_feature_importances(model)

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()