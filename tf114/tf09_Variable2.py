import tensorflow as tf
tf.compat.v1.set_random_seed(12323)

# y = wx + b
import tensorflow as tf
tf.set_random_seed(123)
sess = tf.compat.v1.Session()
# 1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]
test_data = [6, 7, 8]

x_train = tf.placeholder(tf.float32, shape=[None]) # shape=[None] : 1차원 배열, None : 크기가 정해지지 않았다.
y_train = tf.placeholder(tf.float32, shape=[None]) # 

# W = tf.Variable(333, dtype=tf.float32)
# b = tf.Variable(245, dtype=tf.float32)
W = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32) # random_normal : 정규분포 , random_uniform : 균등분포
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32) 


# 2. 모델

hypothesis = x_train * W + b # y = wx + b

# 3-1. 컴파일

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse, reduce_mean : 평균
# loss = tf.matrix_square_root(tf.reduce_mean(tf.square(hypothesis - y))) # rmse, reduce_mean : 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.175) # learning_rate : 학습률, 

train = optimizer.minimize(loss) # loss 최소화
# model.compile(optimizer='sgd', loss='mse')

# 3-2-1. 훈련
with tf.compat.v1.Session() as sess: # with문 : 자동으로 close
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1000

    for step in range(epochs):
        # sess.run(train)
        _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                 feed_dict={x_train:x_train_data, y_train:y_train_data})
        if step % 20 == 0: # % : 나머지
            print(step, loss_val, W_val, b_val)
        if 2.01 >= W_val >= 1.99 and 1.01 >= b_val >= 0.99:
            print('W_val : ', W_val)
            print('b_val : ', b_val)
            print('step : ', step)
            break
    x_test = tf.placeholder(tf.float32, shape=[None])
    y_predict = x_test * W_val + b_val
    print(sess.run(y_predict, feed_dict={x_test:test_data}))

# [5.999973  6.999962  7.9999504]
# 3000 3.0202046e-10 [0.99998873] [4.067214e-05]
#################################################################


# 3-2-2. 훈련
with tf.compat.v1.Session() as sess: # with문 : 자동으로 close
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1000

    for step in range(epochs):
        _ = sess.run(train,
                 feed_dict={x_train:x_train_data, y_train:y_train_data})
        loss_val = loss.eval(session=sess, feed_dict={x_train:x_train_data, y_train:y_train_data})
        W_val = W.eval(session=sess)
        b_val = b.eval(session=sess)
        if step % 20 == 0: # % : 나머지
            print(step, loss_val, W_val, b_val)
        if 2.01 >= W_val >= 1.99 and 1.01 >= b_val >= 0.99:
            print('W_val : ', W_val)
            print('b_val : ', b_val)
            print('step : ', step)
            break
    x_test = tf.placeholder(tf.float32, shape=[None])
    y_predict = x_test * W_val + b_val
    print(y_predict.eval(session=sess, feed_dict={x_test:test_data}))

# [5.999973  6.999962  7.9999504]
# 3000 3.0202046e-10 [0.99998873] [4.067214e-05]
#################################################################

# 3-2-3. 훈련
sess = tf.compat.v1.InteractiveSession() # with문 : 자동으로 close
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 1000

W2 = W.eval()
b2 = b.eval()



for step in range(epochs) :
    _ = sess.run(train,
                 feed_dict={x_train:x_train_data, y_train:y_train_data})
    loss_val = loss.eval(feed_dict={x_train:x_train_data, y_train:y_train_data})
    W_val = W.eval()
    b_val = b.eval()
    if step % 20 == 0: # % : 나머지
        print(step, loss_val, W_val, b_val)
    if 2.01 >= W_val >= 1.99 and 1.01 >= b_val >= 0.99:
        print('W_val : ', W_val)
        print('b_val : ', b_val)
        print('step : ', step)
        break
x_test = tf.placeholder(tf.float32, shape=[None])
y_predict = x_test * W_val + b_val
print(y_predict.eval(feed_dict={x_test:test_data}))

# [5.999973  6.999962  7.9999504]
# 3000 3.0202046e-10 [0.99998873] [4.067214e-05]
#################################################################





