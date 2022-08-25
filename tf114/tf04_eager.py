import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) # executing_eagerly : 즉시실행모드

tf.compat.v1.disable_eager_execution() # 즉시실행모드를 해제한다, v1을 사용한다.

print(tf.executing_eagerly()) # executing_eagerly : 즉시실행모드

hello = tf.constant('hello world!')

sess = tf.compat.v1.Session()

print(sess.run(hello))  # b'hello world!' : b는 byte를 의미한다.