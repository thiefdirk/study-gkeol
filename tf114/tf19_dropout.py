import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random_normal([2, 30]), name='weight1')
b1 = tf.compat.v1.Variable(tf.random_normal([30]), name='bias1')

Hidden_Layer1 = tf.compat.v1.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(30, activation='sigmoid', input_shape=(2,)))

dropout_layers = tf.compat.v1.nn.dropout(Hidden_Layer1, rate=0.3)

print(Hidden_Layer1) #Tensor("Sigmoid:0", shape=(?, 30), dtype=float32)
print(dropout_layers) #Tensor("dropout/mul_1:0", shape=(?, 30), dtype=float32)