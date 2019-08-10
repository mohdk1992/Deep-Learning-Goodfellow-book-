# XOR learning using TensorFlow

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float64, shape=[4, 2], name="x")
y = tf.placeholder(tf.float64, shape=[4, 1], name="y")

m = np.shape(x)[0]
n = np.shape(x)[1]
hidden_s = 2
l_r = 1

Theta1 = tf.cast(tf.Variable(tf.random_normal([2, 2], -1, 1), name="Theta1"), tf.float64)
Theta2 = tf.cast(tf.Variable(tf.random_normal([2, 1], -1, 1), name="Theta2"), tf.float64)

Bias1 = tf.cast(tf.Variable(tf.zeros([2]), name="Bias1"), tf.float64)
Bias2 = tf.cast(tf.Variable(tf.zeros([1]), name="Bias2"), tf.float64)

A2 = tf.sigmoid(tf.matmul(x, Theta1) + Bias1)
H = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = -tf.reduce_mean((y * tf.log(H)) + ((1 - y) * tf.log(1 - H)))
train_step = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

x_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y_data = [
    [0],
    [1],
    [1],
    [0]
]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_data, y: y_data})
    if i % 100 == 0:
        print("Epoch:", i)
        print("Hyp:", sess.run(H, feed_dict={x: x_data, y: y_data}))

