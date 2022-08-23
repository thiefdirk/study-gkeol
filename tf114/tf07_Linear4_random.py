# y = wx + b
import tensorflow as tf
# tf.set_random_seed(123)
sess = tf.compat.v1.Session()
# 1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

# W = tf.Variable(333, dtype=tf.float32)
# b = tf.Variable(245, dtype=tf.float32)
W = tf.Variable(tf.random_uniform([1]), dtype=tf.float32) # random_normal : 정규분포
b = tf.Variable(tf.random_uniform([1]), dtype=tf.float32) # random_normal : 정규분포
init = tf.compat.v1.global_variables_initializer() # 변수 초기화
sess.run(init)
print(sess.run(W), sess.run(b)) # [-1.5080816] [0.9301948]
exit() 


# 2. 모델

hypothesis = x * W + b # y = wx + b

# 3-1. 컴파일

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse, reduce_mean : 평균
# loss = tf.matrix_square_root(tf.reduce_mean(tf.square(hypothesis - y))) # rmse, reduce_mean : 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 경사하강법

train = optimizer.minimize(loss) # loss 최소화
# model.compile(optimizer='sgd', loss='mse')

# 3-2. 훈련
with tf.compat.v1.Session() as sess: # with문 : 자동으로 close
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 3001

    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0: # % : 나머지
            print(step, sess.run(loss), sess.run(W), sess.run(b))




