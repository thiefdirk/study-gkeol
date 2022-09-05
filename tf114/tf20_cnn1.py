import tensorflow as tf
import keras
import numpy as np

tf.compat.v1.set_random_seed(123)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255. # astype : 형변환, float32 : 32비트 실수형
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,64]) # 2,2,1,64 : 2,2 : 커널사이즈, 1 : 컬러, 64 : 필터갯수
#model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # strides : 1,1,1,1 : 1,1 : 스트라이드, # padding='VALID' : 패딩사용안함

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)


