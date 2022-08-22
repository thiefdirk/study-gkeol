import tensorflow as tf
sess = tf.compat.v1.Session()
node1 = tf.constant(3.0, tf.float32) # 상수, float32 : 32비트 실수
node2 = tf.constant(4.0) # 상수
# node3 = node1 + node2 # 덧셈
node3 = tf.add(node1, node2) # 덧셈

print(node3) # Tensor("add:0", shape=(), dtype=float32) : 노드정보

print(sess.run(node3))