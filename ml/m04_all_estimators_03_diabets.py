#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore') # warning 무시
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)

print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 0.0
# print(np.max(x_train))  # 1.0

# print(np.min(x_test))  # 1.0
# print(np.max(x_test))  # 1.0

#2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier') # 분류모델만 추출
allAlgorithms = all_estimators(type_filter='regressor') # 회귀모델만 추출
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

print(allAlgorithms) # 모든 모델을 보여줌
print(len(allAlgorithms)) # 모든 모델의 갯수를 보여줌, 총 갯수는 총 모델의 갯수 + 1, 41


for (name, algorithm) in allAlgorithms: # key, value로 나누어서 보여줌
    try: 
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', acc)
    except: # 모델에 오류가 있을 경우 예외를 발생시킴
        print(name, '은 안나온 놈!!!')
        continue    # 예외가 발생하면 다음 모델으로 넘어가게 하는 코드
# TypeError: __init__() missing 1 required positional argument: 'base_estimator', 이런 에러가 뜸
# 예외처리 해야함

# ARDRegression 의 정답률 : 0.6530571637755074  
# AdaBoostRegressor 의 정답률 : 0.6047549268511743
# BaggingRegressor 의 정답률 : 0.49537492265901795
# BayesianRidge 의 정답률 : 0.6530732433967228  
# CCA 은 안나온 놈!!!
# DecisionTreeRegressor 의 정답률 : 0.005273984862447012
# DummyRegressor 의 정답률 : -0.014400454257647022
# ElasticNet 의 정답률 : -0.005328266814577098  
# ElasticNetCV 의 정답률 : 0.5416195896166865
# ExtraTreeRegressor 의 정답률 : 0.04312920514405627
# ExtraTreesRegressor 의 정답률 : 0.5763416145837457
# GammaRegressor 의 정답률 : -0.007823542660838623
# GaussianProcessRegressor 의 정답률 : -19.89811178013024
# GradientBoostingRegressor 의 정답률 : 0.5188665050919943
# HistGradientBoostingRegressor 의 정답률 : 0.530828319596758
# HuberRegressor 의 정답률 : 0.6577980156650767
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 : 0.5403351561734346
# KernelRidge 의 정답률 : -3.4081572282782817   
# Lars 의 정답률 : 0.6579197606548157
# LarsCV 의 정답률 : 0.6564476184179993
# Lasso 의 정답률 : 0.3785224007954726
# LassoCV 의 정답률 : 0.6586221904334045
# LassoLars 의 정답률 : 0.4154761894569349      
# LassoLarsCV 의 정답률 : 0.6579197606548157
# LassoLarsIC 의 정답률 : 0.6590994359358212    
# LinearRegression 의 정답률 : 0.6579197606548162
# LinearSVR 의 정답률 : -0.4518705170583017     
# MLPRegressor 의 정답률 : -3.227330644315061
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 : 0.15030443671279659
# OrthogonalMatchingPursuit 의 정답률 : 0.4157059575736308
# OrthogonalMatchingPursuitCV 의 정답률 : 0.6538201252570435
# PLSCanonical 은 안나온 놈!!!
# PLSRegression 의 정답률 : 0.6545408930519411  
# PassiveAggressiveRegressor 의 정답률 : 0.5650034697100033
# PoissonRegressor 의 정답률 : 0.39364665773613516
# QuantileRegressor 의 정답률 : -0.06628628382122415
# RANSACRegressor 의 정답률 : 0.25629978581643387
# RadiusNeighborsRegressor 의 정답률 : -0.014400454257647022
# RandomForestRegressor 의 정답률 : 0.5276695422102141
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 : 0.5059105205948112
# RidgeCV 의 정답률 : 0.6432606908795868        
# SGDRegressor 의 정답률 : 0.4844467727737021
# SVR 의 정답률 : 0.14455754772757579
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 : 0.6478364160862691
# TransformedTargetRegressor 의 정답률 : 0.6579197606548162
# TweedieRegressor 의 정답률 : -0.007497539132948372
# VotingRegressor 은 안나온 놈!!!

#3. 컴파일, 훈련

hist = model.fit(x_train, y_train)



end_time = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('acc : ' , results)

# LinearSVR : -0.44838778199913354
# LinearRegression : 0.6579197606548162
# KNeighborsRegressor : 0.5403351561734346
# DecisionTreeRegressor : 0.09974953394661623
# RandomForestRegressor : 0.5413415904728223