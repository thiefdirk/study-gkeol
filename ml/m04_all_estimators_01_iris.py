import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore') # warning 무시

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델
allAlgorithms = all_estimators(type_filter='classifier') # 분류모델만 추출
# [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)

print(allAlgorithms) # 모든 모델을 보여줌
print(len(allAlgorithms)) # 모든 모델의 갯수를 보여줌, 총 갯수는 총 모델의 갯수 + 1, 41


for (name, algorithm) in allAlgorithms: # key, value로 나누어서 보여줌
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 :', accuracy_score(y_test, y_predict))
    except:
        print(name, '은 안나온 놈!!!')
        continue
# TypeError: __init__() missing 1 required positional argument: 'base_estimator', 이런 에러가 뜸
# 예외처리 해야함

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

# LinearSVC 모델의 정확도 :  0.9666666666666667
# LogisticRegression 모델의 정확도 :  1.0
# KNN 모델의 정확도 :  0.9666666666666667
# DecisionTreeClassifier 모델의 정확도 :  0.9333333333333333
# RandomForestClassifier 모델의 정확도 :  0.9333333333333333