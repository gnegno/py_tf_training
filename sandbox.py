import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Those two lines take out the SSE warnings
import tensorflow as tf


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

sess = tf.Session()

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
