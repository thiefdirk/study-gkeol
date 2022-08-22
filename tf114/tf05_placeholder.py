import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # 상수, float32 : 32비트 실수
node2 = tf.constant(4.0) # 상수
node3 = tf.add(node1, node2) # 덧셈

sess = tf.compat.v1.Session()

a = tf.placeholder(tf.float32) # placeholder : 변수
b = tf.placeholder(tf.float32)

add_node = a + b # 덧셈

print(sess.run(add_node, feed_dict={a:3, b:4.5})) # 7.5
print(sess.run(add_node, feed_dict={a:[1,3], b:[2,4]})) # [3. 7.]

mul_node = a * b # 곱셈

print(sess.run(mul_node, feed_dict={a:3, b:4.5})) # 13.5
print(sess.run(mul_node, feed_dict={a:[1,3], b:[2,4]})) # [ 2. 12.]

matmul_node = tf.matmul(a, b) # 행렬곱

print(sess.run(matmul_node, feed_dict={a:[[1,2], [3,4]], b:[[1,2], [3,4]]})) # [[ 7. 10.][15. 22.]]

