from tabnanny import verbose
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)
# print(datasets.feature_names)
# print(datasets.DESCR)

start_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )
end_time = time.time()

#2. 모델구성

model = Sequential()
model.add(Dense(200, input_dim=10))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(180))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=100,verbose=1,validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)


# print('==========================')
# print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001388A0478B0>
# print('==========================')
# print(hist.history)
# print('==========================')
# print(hist.history['loss'])
# print('==========================')
# print(hist.history['val_loss'])

print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', label='loss', color='red')
# plt.plot(hist.history['val_loss'], marker='.', label='val_loss', color='blue')
# plt.grid()
# plt.title('당뇨')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()
# r2 0.62 이상
# loss :  2452.336669921875
# r2스코어 :  0.6286149246878252
##################val전후#################
# loss :  2211.544189453125
# r2스코어 :  0.6650808519754101
##################EarlyStopping전후#################
# loss :  2170.21484375
# r2스코어 :  0.6713398311092679