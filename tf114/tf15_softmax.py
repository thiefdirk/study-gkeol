import numpy as np
import tensorflow as tf
tf.set_random_seed(123)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# 2. 모델

x = tf.placeholder(tf.float32, shape=[None, 4])

W = tf.Variable(tf.random_normal([4, 3]), name='weight')

b = tf.Variable(tf.random_normal([1, 3]), name='bias')

y = tf.placeholder(tf.float32, shape=[None, 3])

hypothsis = tf.nn.softmax(tf.matmul(x, W) + b) # tf.nn : 신경망 관련 함수

# 3. 훈련

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothsis), axis=1)) # reduce_sum : 합계, reduce_mean : 평균