import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],  # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]] #(5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([3, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00008)
train = optimizer.minimize(loss)

#3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 500001
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if epochs % 20 == 0 :
        print(epochs, 'loss : ', cost_val, '\n', hy_val)
        
sess.close()

y_pred = hy_val


print('예측값 : ', y_pred)

r2 = r2_score(y_data, y_pred)
print('R2 : ', r2)

mae = mean_absolute_error(y_data, y_pred)
print('MAE : ', mae)
