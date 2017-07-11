import tensorflow as tf

class Model():
    def __init__(self, learning_rate, mnist, sess):
        self.sess = sess
        self.mnist = mnist
        self.x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.test_sum_cross_entropy = tf.summary.scalar("test.cross_entropy",
                                            self.cross_entropy)
        self.test_sum_acc = tf.summary.scalar("test.accuracy", self.accuracy)
        tf.global_variables_initializer().run()

    def train(self, minibatch_size):
        mnist = self.mnist
        batch_xs, batch_ys = mnist.train.next_batch(minibatch_size)
        # take a train step:
        self.sess.run(self.train_step, feed_dict={self.x: batch_xs,
                                                 self.y_: batch_ys})