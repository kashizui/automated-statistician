import tensorflow as tf
import time

a = tf.Variable(0.)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# The slow method that constructs a new ops each time
# The new ops in this case being tf.assign
t = time.time()
for _ in xrange(1000):
    sess.run(tf.assign(a, 1.))
print "Slow_Method_Duration: {0:.5f}".format(time.time() - t)

# The fast method that does all the tf ops beforehand
hold = tf.placeholder(tf.float32, [])
ass = tf.assign(a, hold)
t = time.time()
for _ in xrange(1000):
    sess.run(ass, feed_dict={hold : 1.})
print "Fast_Method_Duration: {0:.5f}".format(time.time() - t)
