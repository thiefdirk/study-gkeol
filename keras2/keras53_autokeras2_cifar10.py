import autokeras as ak
print(ak.__version__)
import tensorflow as tf
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,)

#2. 모델 구성
model = ak.ImageClassifier( # ImageClassifier는 이미지를 입력받아 이미지를 추론하는 모델을 만든다.,overwrite=True는 이미 존재하는 모델을 덮어쓰기 위해서 사용한다.
                           overwrite=True,
                           max_trials=2) 

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs=5)

#4. 평가, 예측
y_pred = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print('loss, accuracy_score : ', results)
print('time : ', round(time.time() - start, 4))

# loss, accuracy_score :  [0.8731628656387329, 
# 0.6966000199317932]
# time :  2785.5388