# DNN으로 구성
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
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

#hidden layer 1
w1 = tf.compat.v1.get_variable('weight1', shape=[2,2,1,16]) # 2,2,1,64 : 2,2 : 커널사이즈, 1 : 컬러, 64 : 필터갯수
#model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.random_normal([3,3,16,15], name='weight2'))

#output layer
w3 = tf.compat.v1.Variable(tf.random_normal([2535,10], name='weight3'))


L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # strides : 1,1,1,1 : 1,1 : 스트라이드, # padding='VALID' : 패딩사용안함
dropout_layers1 = tf.compat.v1.nn.dropout(L1, rate=0.3)
L2 = tf.nn.conv2d(dropout_layers1, w2, strides=[1,2,2,1], padding='VALID') # strides : 1,1,1,1 : 1,1 : 스트라이드, # padding='VALID' : 패딩사용안함
dropout_layers2 = tf.compat.v1.nn.dropout(L2, rate=0.2)
F1 = tf.compat.v1.layers.flatten(dropout_layers2)
print(L1) # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)
print(L2) # Tensor("Conv2D_1:0", shape=(?, 8, 13, 15), dtype=float32)
print(F1.shape) # Tensor("Conv2D_1:0", shape=(?, 8, 13, 15), dtype=float32)
hypothesis = tf.nn.softmax(tf.matmul(F1, w3)) # tf.nn : 신경망 관련 함수



# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.matmul(F1, w3), labels=y)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 31
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 10 == 0 :
        print(epochs, 'loss : ', cost_val, '\n')
        
# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)

sess.close()
# acc :  0.5455
