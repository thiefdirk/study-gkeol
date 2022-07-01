from tabnanny import verbose
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_boston
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)

print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

print('==========================')
print(hist) #<tensorflow.python.keras.callbacks.History object at 0x000001388A0478B0>
print('==========================')
print(hist.history)
print('==========================')
print(hist.history['loss'])
print('==========================')
print(hist.history['val_loss'])

print("걸린시간 : ", end_time)

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', label='loss', color='red')
plt.plot(hist.history['val_loss'], marker='.', label='val_loss', color='blue')
plt.grid()
plt.title('안결바보')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()