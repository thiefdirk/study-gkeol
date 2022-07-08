from warnings import filters
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
# model.add(Dense(units=10, input_shape = (3,)))         #  (batch_size, input_dim)             input_shape = (10, 10, 3)
# model.summary()
# (input_dim + bias) * units = summary Param # (Dense 모델)


model.add(Conv2D(filters = 64, kernel_size=(3,3), # kernel_size = 이미지 분석을위해 2x2로 잘라서 분석하겠다~
                 padding='same', # padding : 커널 사이즈대로 자르다보면 가생이는 중복되서 분석을 못해주기때문에 행렬을 키워주는것, 패딩을 입혀준다? 이런 너낌
                 input_shape=(28,28,1))) #  (batch_size, rows, columns, channels)            conv2d : model.add input_shape= (x, y, z) x=가로 픽셀 y=세로픽셀 z= 컬러 흑백
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트 값
                 activation='relu'))
model.add(Flatten())  # (N, 5408)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.summary()
# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈