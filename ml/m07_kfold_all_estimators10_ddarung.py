# Dacon 따릉이 문제풀이
import pandas as pd
from pandas import DataFrame 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np

# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

n_splits = 10

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 불필요한 warning 제거

allAlgorithms = all_estimators(type_filter='regressor')
#allAlgorithms = all_estimators(type_filter='classifier')

#('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

#print('allAlgorithms : ', allAlgorithms)
#print('모델갯수 : ', len(allAlgorithms)) # 모델갯수 :  41
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm()
        # model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print('Model Name : ', name)
        print('ACC : ', scores) 
        print('cross_val_score : ', round(np.mean(scores), 4))
        
        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
        #print(y_predict)
        
        #y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        # print(name, '의 정답률 : ', acc )
    except:
        # continue
        print(name, '은 실행되지 않는다.')

'''
Model Name :  ARDRegression
ACC :  [0.55192929 0.60145066 0.56701428 0.56954619 0.45872221 0.52979943
 0.61470581 0.55947748 0.72297388 0.52524782]
cross_val_score :  0.5701
Model Name :  AdaBoostRegressor
ACC :  [0.59696695 0.57385436 0.6666106  0.58987097 0.48880941 0.51382998
 0.6223837  0.62957347 0.60732814 0.41399134]
cross_val_score :  0.5703
Model Name :  BaggingRegressor
ACC :  [0.75608743 0.69354553 0.79574791 0.6905549  0.69077915 0.76385767
 0.78115405 0.7675511  0.79046442 0.66505593]
cross_val_score :  0.7395
Model Name :  BayesianRidge
ACC :  [0.55265583 0.60449011 0.56894042 0.57221978 0.45985341 0.52812702
 0.61229961 0.56108062 0.71336481 0.52310614]
cross_val_score :  0.5696
Model Name :  CCA
ACC :  [-0.04407269  0.25114809  0.42921369  0.36149005 -0.21538048  0.06967343
  0.26587743  0.22833816  0.40079567 -0.14628252]
cross_val_score :  0.1601
Model Name :  DecisionTreeRegressor
ACC :  [0.57663953 0.44719994 0.63303241 0.58455235 0.54304671 0.45573371
 0.56779404 0.65857296 0.6792177  0.50880623]
cross_val_score :  0.5655
Model Name :  DummyRegressor
ACC :  [-0.01069899 -0.06307062 -0.13849286 -0.003278   -0.00571261 -0.00681972
 -0.00358828 -0.02449918 -0.01400969 -0.03079914]
cross_val_score :  -0.0301
Model Name :  ElasticNet
ACC :  [0.50800713 0.52776704 0.46647628 0.49598141 0.42978614 0.46989536
 0.5144925  0.48678027 0.59798961 0.49762533]
cross_val_score :  0.4995
Model Name :  ElasticNetCV
ACC :  [0.54931744 0.60426279 0.56368166 0.57271729 0.47392468 0.52596774
 0.59501722 0.55529517 0.69169065 0.51727737]
cross_val_score :  0.5649
Model Name :  ExtraTreeRegressor
ACC :  [0.50480768 0.27964454 0.65269464 0.53869314 0.54114268 0.48734694
 0.56704421 0.77166834 0.50585162 0.44951748]
cross_val_score :  0.5298
Model Name :  ExtraTreesRegressor
ACC :  [0.7581161  0.75050945 0.82143206 0.75888035 0.72788526 0.77002236
 0.82387658 0.78598767 0.81917309 0.75096531]
cross_val_score :  0.7767
Model Name :  GammaRegressor
ACC :  [0.29707001 0.38903625 0.34705407 0.35077427 0.28921166 0.34936068
 0.35559028 0.29689907 0.38017455 0.29344542]
cross_val_score :  0.3349
Model Name :  GaussianProcessRegressor
ACC :  [-0.08885376  0.0780209   0.07970742 -0.11544322 -0.32309663 -0.04913496
  0.24160183  0.30045562  0.47738981 -0.4183112 ]
cross_val_score :  0.0182
Model Name :  GradientBoostingRegressor
ACC :  [0.75037178 0.71993287 0.77894695 0.702497   0.71135538 0.74171265
 0.80005741 0.79905812 0.82709149 0.71486507]
cross_val_score :  0.7546
Model Name :  HistGradientBoostingRegressor
ACC :  [0.79031297 0.73386868 0.77822945 0.74689045 0.77397088 0.75118832
 0.79397889 0.79846149 0.80609736 0.74873505]
cross_val_score :  0.7722
Model Name :  HuberRegressor
ACC :  [0.55034953 0.57537486 0.51120495 0.56931913 0.49088122 0.52918897
 0.58275341 0.55378339 0.70173777 0.53498884]
cross_val_score :  0.56
Model Name :  IsotonicRegression
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
IsotonicRegression 은 실행되지 않는다.
Model Name :  KNeighborsRegressor
ACC :  [0.56815032 0.68102656 0.59601492 0.62934045 0.59491752 0.53825383
 0.62010579 0.56576452 0.67217156 0.55485032]
cross_val_score :  0.6021
Model Name :  KernelRidge
ACC :  [-0.21501572 -0.29957277 -0.48282168 -0.2720295  -0.35881124 -0.26716174
 -0.16250001 -0.32389484 -0.12673972 -0.49185761]
cross_val_score :  -0.3
Model Name :  Lars
ACC :  [0.55358213 0.60284151 0.56888474 0.57093716 0.45476452 0.52751734
 0.61502885 0.56067036 0.71764972 0.5249708 ]
cross_val_score :  0.5697
Model Name :  LarsCV
ACC :  [0.55454321 0.60262173 0.55661677 0.57071814 0.46756319 0.52792654
 0.61474408 0.56098588 0.71764972 0.52578649]
cross_val_score :  0.5699
Model Name :  Lasso
ACC :  [0.5442208  0.60339702 0.55983695 0.57190884 0.48059379 0.53264107
 0.59583188 0.55714008 0.69303333 0.50487414]
cross_val_score :  0.5643
Model Name :  LassoCV
ACC :  [0.55330585 0.60314246 0.56813823 0.57118114 0.45647217 0.52792571
 0.61400616 0.5608053  0.71711476 0.52456327]
cross_val_score :  0.5697
Model Name :  LassoLars
ACC :  [0.34701231 0.29195083 0.18307592 0.29778316 0.32857744 0.3156284
 0.32419019 0.30068343 0.34025365 0.31704998]
cross_val_score :  0.3046
Model Name :  LassoLarsCV
ACC :  [0.55454321 0.60262173 0.56748384 0.57070662 0.4560004  0.52792654
 0.61474408 0.56098588 0.71764972 0.52578649]
cross_val_score :  0.5698
Model Name :  LassoLarsIC
ACC :  [0.55395742 0.60283796 0.56427118 0.57106314 0.46859057 0.5283444
 0.61461598 0.56069118 0.71912022 0.52519082]
cross_val_score :  0.5709
Model Name :  LinearRegression
ACC :  [0.55358213 0.60284151 0.56888474 0.57093716 0.45476452 0.52751734
 0.61502885 0.56067036 0.71764972 0.5249708 ]
cross_val_score :  0.5697
Model Name :  LinearSVR
ACC :  [0.4913542  0.48564056 0.37178579 0.50344145 0.52314538 0.46963116
 0.47191208 0.48779967 0.61148841 0.49277397]
cross_val_score :  0.4909
Model Name :  MLPRegressor
ACC :  [0.47854197 0.51103792 0.48589892 0.47128704 0.40461472 0.40714304
 0.48994389 0.49903808 0.63719277 0.52668968]
cross_val_score :  0.4911
MultiOutputRegressor 은 실행되지 않는다.
Model Name :  MultiTaskElasticNet
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskElasticNet 은 실행되지 않는다.
Model Name :  MultiTaskElasticNetCV
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskElasticNetCV 은 실행되지 않는다.
Model Name :  MultiTaskLasso
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskLasso 은 실행되지 않는다.
Model Name :  MultiTaskLassoCV
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
MultiTaskLassoCV 은 실행되지 않는다.
Model Name :  NuSVR
ACC :  [0.39526307 0.35780914 0.25349243 0.39545856 0.40320087 0.37623607
 0.38410293 0.38553521 0.43436842 0.42679148]
cross_val_score :  0.3812
Model Name :  OrthogonalMatchingPursuit
ACC :  [0.37431875 0.35964579 0.2612898  0.36081413 0.23009344 0.35045519
 0.40895875 0.33510797 0.42366781 0.26015537]
cross_val_score :  0.3365
Model Name :  OrthogonalMatchingPursuitCV
ACC :  [0.55644271 0.58752536 0.54601322 0.55046807 0.44380775 0.49942857
 0.60836867 0.5505755  0.6988597  0.49425976]
cross_val_score :  0.5536
Model Name :  PLSCanonical
ACC :  [-0.732338   -0.382492    0.05701853  0.03024751 -0.90592318 -0.59820164
 -0.40006069 -0.34807833 -0.27377957 -1.02725519]
cross_val_score :  -0.4581
Model Name :  PLSRegression
ACC :  [0.54925106 0.60516062 0.56939033 0.57569327 0.44647298 0.52773519
 0.6045289  0.56526438 0.70377481 0.51944407]
cross_val_score :  0.5667
Model Name :  PassiveAggressiveRegressor
ACC :  [0.51323803 0.50995963 0.44699548 0.56259056 0.49780703 0.48214816
 0.54184532 0.48009118 0.67817    0.53226702]
cross_val_score :  0.5245
Model Name :  PoissonRegressor
ACC :  [0.55642518 0.63107006 0.61467998 0.61199191 0.52115464 0.58273185
 0.6189005  0.57427151 0.73801724 0.51499665]
cross_val_score :  0.5964
Model Name :  RANSACRegressor
ACC :  [0.50987539 0.39080786 0.28361743 0.51694286 0.52352418 0.43210396
 0.38125792 0.49093237 0.62623034 0.4076631 ]
cross_val_score :  0.4563
Model Name :  RadiusNeighborsRegressor
ACC :  [nan nan nan nan nan nan nan nan nan nan]
cross_val_score :  nan
Model Name :  RandomForestRegressor
ACC :  [0.78921842 0.72061564 0.78829823 0.74616997 0.75556389 0.72637864
 0.79852169 0.79595799 0.809839   0.76867302]
cross_val_score :  0.7699
RegressorChain 은 실행되지 않는다.
Model Name :  Ridge
ACC :  [0.55332917 0.603373   0.5689262  0.57134102 0.45640642 0.52772487
 0.61435424 0.56084334 0.71663906 0.52447912]
cross_val_score :  0.5697
Model Name :  RidgeCV
ACC :  [0.55332917 0.603373   0.5689262  0.57134102 0.45640642 0.52772487
 0.61435424 0.56084334 0.71663906 0.52447912]
cross_val_score :  0.5697
Model Name :  SGDRegressor
ACC :  [0.54618403 0.6067807  0.56994646 0.57593325 0.46679098 0.52974747
 0.60552214 0.5575532  0.70952758 0.51181065]
cross_val_score :  0.568
Model Name :  SVR
ACC :  [0.40566615 0.35382588 0.2427564  0.41631835 0.42496134 0.39310576
 0.38995454 0.40988231 0.45672637 0.44557616]
cross_val_score :  0.3939
StackingRegressor 은 실행되지 않는다.
Model Name :  TheilSenRegressor
ACC :  [0.53269874 0.60330784 0.55054887 0.57167136 0.49349099 0.51380842
 0.57026179 0.55548304 0.68406119 0.4780588 ]
cross_val_score :  0.5553
Model Name :  TransformedTargetRegressor
ACC :  [0.55358213 0.60284151 0.56888474 0.57093716 0.45476452 0.52751734
 0.61502885 0.56067036 0.71764972 0.5249708 ]
cross_val_score :  0.5697
Model Name :  TweedieRegressor
ACC :  [0.45821462 0.46075227 0.38890972 0.433366   0.38775976 0.41898181
 0.45565278 0.42963162 0.52602902 0.45296147]
cross_val_score :  0.4412
VotingRegressor 은 실행되지 않는다.
'''