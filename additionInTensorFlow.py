import tensorflow as tf

a = tf.placeholder('int32')
b = tf.placeholder('int32')
c = a + b
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
answer = sess.run(c, feed_dict={a:1, b:2})
print(answer)