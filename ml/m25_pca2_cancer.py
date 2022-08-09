import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
print(sk.__version__) # 1.1.1
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)
# print(range(len(pd.DataFrame(x.columns)))) # range(30)
x_columns = pd.DataFrame(x)

# pca = PCA(n_components=12) # n_components : 주요하지 않은 변수를 제거하고 싶은 개수를 지정한다.
# x = pca.fit_transform(x) # x를 pca로 변환한다.
# print(x.shape) # (506, 2)

for i in range(len(x_columns)) :
    x = datasets.data
    y = datasets.target
    
    PCA(n_components=i).fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)
#2. 모델
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
# model = XGBRegressor()
    model = RandomForestRegressor()
#3. 훈련
    model.fit(x_train, y_train) #, eval_metric='error')
#4. 평가 및 예측
    result = model.score(x_test, y_test)
    print(str(i+1)+'회 model.score : ', result)

# 1회 model.score :  0.8547937854994988
# 2회 model.score :  0.8463799532241897
# 3회 model.score :  0.8446469094553959
# 4회 model.score :  0.8348809221516872
# 5회 model.score :  0.8588731039091212
# 6회 model.score :  0.8504173738723688
# 7회 model.score :  0.8472331440026728
# 8회 model.score :  0.8588273972602739
# 9회 model.score :  0.8530112261944537
# 10회 model.score :  0.8511448713665218
# 11회 model.score :  0.8577304376879384
# 12회 model.score :  0.8526455730036752
# 13회 model.score :  0.8350142332108251
# 14회 model.score :  0.8511144002672903
# 15회 model.score :  0.8530264617440695
# 16회 model.score :  0.8488900100233878
# 17회 model.score :  0.8454848646842632
# 18회 model.score :  0.8451725359171399
# 19회 model.score :  0.8418283327764784
# 20회 model.score :  0.8505125960574673
# 21회 model.score :  0.851072502505847
# 22회 model.score :  0.8527560307383895
# 23회 model.score :  0.855243234213164
# 24회 model.score :  0.8412341463414633
# 25회 model.score :  0.8534492482459071
# 26회 model.score :  0.83164717674574
# 27회 model.score :  0.8535520882058135
# 28회 model.score :  0.8496632141663882
# 29회 model.score :  0.8569343802205145
# 30회 model.score :  0.8575514199799532
# 31회 model.score :  0.860347143334447