import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

tf.compat.v1.set_random_seed(123)

#1. 데이터

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] # 

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2]) # shape=[None, 2] : 2차원 배열, None : 크기가 정해지지 않았다.
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # shape=[None, 1] : 2차원 배열, None : 크기가 정해지지 않았다.

w= tf.Variable(tf.compat.v1.random_normal([2,1]), name='weight') # shape=[2,1] : 2차원 배열
b= tf.Variable(tf.compat.v1.random_normal([1]), name='bias') # shape=[1] : 1차원 배열

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # sigmoid : 0~1 사이의 값으로 변환

# loss = tf.reduce_mean(tf.square(hypothesis - y))
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis)) # binary_crossentropy


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epoch = 2000
for epochs in range(epoch) :
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train],
                                   feed_dict={x: x_data, y: y_data})
    if epochs % 20 == 0 :
        print(epochs, 'loss : ', cost_val, '\n', hy_val)
        




y_pred = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32)) # 0.5보다 크면 1, 작으면 0, cast: 형변환


print('예측값 : ', y_pred)

acc = accuracy_score(y_data, y_pred)
print('acc : ', acc)

sess.close()