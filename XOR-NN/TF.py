# Test

import tensorflow as tf

with tf.Session() as sess:

    a = tf.constant(15, name = "a")
    b = tf.constant(5, name = "b")

    prod = tf.multiply(a, b, name = "Multi")
    # noinspection PyShadowingBuiltins
    sum = tf.add(a, b, name = "Add")
    res = tf.divide(prod, sum, name = "Divide")

    out = sess.run(res)
    print(out)
