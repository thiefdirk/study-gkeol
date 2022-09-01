from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import numpy
epsilon=numpy.finfo('float').eps
tf.set_random_seed(66)

# 1. 데이터
sess = tf.compat.v1.Session()
datasets = load_boston()
# x_data = pd.DataFrame(datasets.data, columns=datasets.feature_names)
# y_data = pd.DataFrame(datasets.target, columns=['target'])
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)
# print(x_data) # (569, 30)
# print(y_data) # (569, 1)
# print(x_data.dtype) # float64
# print(y_data.dtype) # int32
# change type
# x_data = x_data.astype('float32')
# y_data = y_data.astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=72)
sess.run(tf.compat.v1.global_variables_initializer())
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x=tf.compat.v1.placeholder(tf.float32, shape=[None,13])
y=tf.compat.v1.placeholder(tf.float32, shape=[None,1])


#hidden layer 1
w1 = tf.compat.v1.Variable(tf.random_normal([13,20], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random_normal([20], name='bias1'))

#hidden layer 2
w2 = tf.compat.v1.Variable(tf.random_normal([20,10], name='weight1'))
b2 = tf.compat.v1.Variable(tf.random_normal([10], name='bias1'))

#output layer
w3 = tf.compat.v1.Variable(tf.random_normal([10,1], name='weight2'))
b3 = tf.compat.v1.Variable(tf.random_normal([1], name='bias2'))


hidden_layer1 = tf.matmul(x, w1) + b1 # sigmoid : 0~1 사이의 값으로 변환
hidden_layer2 = tf.matmul(hidden_layer1, w2) + b2 # sigmoid : 0~1 사이의 값으로 변환
hypothesis = tf.matmul(hidden_layer2, w3) + b3 # sigmoid : 0~1 사이의 값으로 변환
# loss = mean_squared_error(y, hypothesis)
loss = tf.reduce_mean(tf.square(hypothesis - y))
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.matmul(x, w) + b)
# log1p : log(1+x) 계산
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2001
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_train, y: y_train})
    if epochs % 100 == 0 :
        print(epochs, 'loss : ', cost_val, '\n')
        
# 4. 평가, 예측
y_pred = sess.run(hypothesis, feed_dict={x: x_test})

r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

sess.close()

# r2 :  0.695031019972476