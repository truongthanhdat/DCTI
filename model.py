import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class DCTI:
    def __init__(self, _num_class=10, _learning_rate=0.001, _regularization_rate=0.0005, _batch_size=32, _num_epoch=200):
        self.num_class = _num_class
        self.batch_size = _batch_size
        self.num_epoch = _num_epoch
        self.learning_rate = _learning_rate
        self.regularization_rate = _regularization_rate
        self.net = self.build_network()
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_network(self):
        self.input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        self.label = tf.placeholder(shape=[None, self.num_class], dtype=tf.float32)
        self.regularizer = slim.l2_regularizer(self.regularization_rate)

        # First Phase 32x32
        net = slim.conv2d(self.input, 64, [3, 3], scope='conv1_1',  weights_regularizer=self.regularizer)
        net = slim.conv2d(net, 64, [3, 3], scope='conv1_2', weights_regularizer=self.regularizer)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Second Phase 16x16
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_1', weights_regularizer=self.regularizer)
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_2', weights_regularizer=self.regularizer)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Third Phase 8x8
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_1', weights_regularizer=self.regularizer)
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_2', weights_regularizer=self.regularizer)
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_3', weights_regularizer=self.regularizer)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        #Fourth Phase 4x4
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_1', weights_regularizer=self.regularizer)
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_2', weights_regularizer=self.regularizer)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #Fifth Phase 2x2
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1', weights_regularizer=self.regularizer)
        net = slim.avg_pool2d(net, [2, 2], scope='pool5')

        # Fully Connected
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, self.num_class, activation_fn=None, scope='fc', weights_regularizer=self.regularizer)

        #Loss function
        self.softmax_loss = slim.losses.softmax_cross_entropy(net, self.label)
        self.loss =  slim.losses.get_total_loss(add_regularization_losses=False)
        self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.label, axis=1), predictions=tf.argmax(net, axis=1))


    def forward(self, sess, images):
        predict = sess.run(self.net, feed_dict = {self.input: images})
        return predict

    def backward(self, sess, images, labels):
        _, summary, loss, accuracy = sess.run([self.trainer, self.summary_op, self.loss, self.accuracy], feed_dict={self.input: images, self.label: labels})
        self.summary_writer.add_summary(summary, self.counter)
        return loss, accuracy

    def train_epoch(self, sess, images, labels):
        length = len(images)
        first = 0
        loss = 0.0
        accuracy = 0.0
        iter = 0
        while (first < length):
            iter = iter + 1
            self.counter = self.counter + 1
            last = first + self.batch_size
            if (last > length):
                last = length
            l, a = self.backward(sess, images[first:last], labels[first:last])
            loss = loss + l * (last - first)
            accuracy = accuracy + a[0] * (last - first)
            print '\tIteration %d: Loss: %f Accuracy %f' % (iter, l, a[0] * 100) + '%'
            first = last

        return loss / float(length), accuracy / float(length)

    def train(self, sess, images, labels):

        tf.summary.scalar("Loss", self.loss)
        tf.summary.tensor_summary("Accuracy", self.accuracy)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())

        saver = tf.train.Saver()
        self.counter = 0
        for i in xrange(1, self.num_epoch + 1):
            loss, accuracy = self.train_epoch(sess, images, labels)
            print 'Epoch %d: Loss: %f Accuracy %f' % (i, loss, accuracy * 100) + '%'
            saver.save(sess, 'output/cifar_%d_model' % self.num_class)


def parse_data_cifar10():
    import pickle
    data = []
    labels = []
    for i in xrange(1, 6):
        dict = pickle.load(open('data/data_batch_%d' % i, 'rb'))
        length = len(dict['labels'])
        arr = np.array(dict['data'], dtype=np.float32) / 255
        lab = np.zeros((length, 10), dtype=np.float32)
        lab[np.arange(length), np.array(dict['labels'], dtype=np.int)] = 1.0
        data.append(arr)
        labels.append(lab)
    data = np.vstack(data)
    data = data.reshape((-1, 32, 32, 3))
    labels = np.vstack(labels)
    return data, labels

if __name__ == '__main__':
    data, labels = parse_data_cifar10()
    net = DCTI(_batch_size=256)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    net.train(sess, data, labels)



