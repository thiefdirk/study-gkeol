import tensorflow as tf
sess = tf.compat.v1.Session()
node1 = tf.constant(2.0, tf.float32) # 상수, float32 : 32비트 실수
node2 = tf.constant(3.0) # 상수

# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6

# node3 = tf.add(node1, node2) # 덧셈
# node4 = tf.subtract(node1, node2) # 뺄셈
# node5 = tf.multiply(node1, node2) # 곱셈
# node6 = tf.divide(node1, node2) # 나눗셈

node3 = node1 + node2 # 덧셈
node4 = node1 - node2 # 뺄셈
node5 = node1 * node2 # 곱셈
node6 = node1 / node2 # 나눗셈

print(sess.run(node3))
print(sess.run(node4))
print(sess.run(node5))
print(sess.run(node6))

