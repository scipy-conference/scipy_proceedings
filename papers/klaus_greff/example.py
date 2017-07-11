import tensorflow as tf
import sacred
from model import Model
from tensorflow.examples.tutorials.mnist\
    import input_data


ex = sacred.Experiment("MNIST")

@ex.config
def config():
    steps = 500
    learning_rate = 0.5
    minibatch_size = 100
    log_dir = "./log/default"


@ex.automain
@sacred.stflow.LogFileWriter(ex)
def experiment(_run, steps, learning_rate,
                minibatch_size, log_dir):
    mnist = input_data.read_data_sets("MNIST_data/",
                                      one_hot=True)
    sess = tf.InteractiveSession()
    nn_model = Model(learning_rate, mnist, sess)
    summary_writer = tf.summary.FileWriter(log_dir)
    test_summary = tf.summary.merge(
                    [nn_model.test_sum_cross_entropy,
                    nn_model.test_sum_acc])
    for _ in range(steps):
        nn_model.train(minibatch_size)
        # evaluate on test
        summary, val_crentr, val_acc = \
            sess.run((test_summary,
                      nn_model.cross_entropy,
                      nn_model.accuracy),
                     feed_dict=
                     {nn_model.x: mnist.test.images,
                      nn_model.y_: mnist.test.labels})
        summary_writer.add_summary(summary, steps)
        _run.log_scalar("test.cross_entropy",
                        float(val_crentr))
        # We can also specify the step number directly
        _run.log_scalar("test.accuracy",
                        float(val_acc), steps)

    return float(val_acc)

