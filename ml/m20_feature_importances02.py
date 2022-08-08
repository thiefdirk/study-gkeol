import numpy as np
from sklearn.datasets import load_iris, load_diabetes

# 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

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
print('accuracy_score:', r2)

print("===================================")
print(model,':',model.feature_importances_)


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