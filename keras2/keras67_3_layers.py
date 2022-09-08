from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.layers[2].trainable = False # 훈련을 동결한다. (가중치를 동결한다.)

# for layer in model.layers:
#     layer.trainable = False
    
model.summary()

# Total params: 17
# Trainable params: 17
# Non-trainable params: 0

# Total params: 17
# Trainable params: 11
# Non-trainable params: 6