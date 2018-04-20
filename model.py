import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys

FLOAT_TYPE = tf.float32

class DCTI:
    def __init__(self, num_class=10, learning_rate=0.001, regularization_rate=0.0005, batch_size=256, num_epoch=200, logdir='logs', model_dir='model'):
        self.num_class = num_class
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate

        self.images = tf.placeholder(shape=[None, 32, 32, 3], dtype=FLOAT_TYPE)
        self.labels = tf.placeholder(shape=[None, 10], dtype=FLOAT_TYPE)

        with slim.arg_scope([slim.conv2d, slim.fully_connected], \
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=0)):
            self.net=  self.build_network(is_training=True)

        self.logdir = logdir
        self.model_dir = model_dir

    def build_network(self, is_training=True):
        # First Phase 32x32
        net = slim.conv2d(self.images, 64, [3, 3], scope='conv1_1')     #Conv2D: 3x3x64
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.dropout(net, keep_prob=0.7)                          #Dropout: keep_prob = 0.7
        net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')             #Conv2D: 3x3x64
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool1')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.7)                          #Dropout: keep_prob = 0.7

        # Second Phase 16x16
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')            #Conv2D: 3x3x128
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.dropout(net, keep_prob=0.7)                          #Dropout: keep_prob = 0.7
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')            #Conv2D: 3x3x128
        net = slim.batch_norm(net, is_training=is_training)             #Barch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool2')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.7)                          #Dropout: keep_prob = 0.7

        # Third Phase 8x8
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')            #Conv2D: 3x3x256
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.dropout(net, keep_prob=0.6)                          #Dropout: keep_prob = 0.6
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')            #Conv2D: 3x3x256
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.dropout(net, keep_prob=0.6)                          #Dropout: keep_prob = 0.6
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')            #Conv2D: 3x3x256
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool3')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.6)                          #Dropout: keep_prob = 0.6

        #Fourth Phase 4x4
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')            #Conv2D: 3x3x512
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.dropout(net, keep_prob=0.6)                          #Dropout: keep_prob = 0.6
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')            #Conv2D: 3x3x512
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool4')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.6)                          #Dropout: keep_prob = 0.6

        #Fifth Phase 2x2
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')            #Conv2D: 3x3x512
        net = slim.batch_norm(net, is_training=is_training)             #Batch Norm
        net = slim.avg_pool2d(net, [2, 2], scope='pool5')               #Global Average Pooling
        net = slim.dropout(net, keep_prob=0.5)                          #Dropout: keep_prob = 0.5

        # Fully Connected
        net = slim.flatten(net, scope='flatten')                                            #Flatten
        net = slim.fully_connected(net, self.num_class, activation_fn=None, scope='fc')     #Fully Connected: 512 >> 10
        net = slim.batch_norm(net, is_training=is_training)                                 #Batch Norm

        return net


    def backward(self, sess, images, labels):
        _, loss, accuracy, summary = sess.run([self.optimizer, self.loss, self.accuracy, self.summary_op], feed_dict={self.images: images, self.labels: labels})
        self.counter = self.counter + 1
        self.summary_writer.add_summary(summary, self.counter)
        return loss, accuracy

    def epoch(self, sess, images, labels):
        total_loss = 0.0
        total_accuracy = 0.0
        first = 0
        length = len(images)
        iter = 0
        while (first < length):
            last = min(length, first + self.batch_size)
            loss, accuracy = self.backward(sess, images[first:last], labels[first:last])
            total_loss = total_loss + loss * (last - first)
            total_accuracy = total_accuracy + accuracy * (last - first)
            first = last
            iter = iter + 1
            sys.stdout.write("\033[F") # Cursor up one line
            print '\tIteration %d: Loss: %.5f Accuracy: %.2f' % (iter, loss, accuracy * 100) + '%'

        return total_loss / float(length), total_accuracy / float(length)

    def train(self, sess, images, labels):
        #Softmax loss
        self.softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.net, labels=self.labels))
        #Regularization loss
        self.weights = tf.trainable_variables()
        self.regularization_loss = tf.add_n([ tf.nn.l2_loss(weight) for weight in self.weights ]) * self.regularization_rate
        #Total loss
        self.loss = self.softmax_loss + self.regularization_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.net, axis=1), tf.argmax(self.labels, axis=1)), FLOAT_TYPE))

        #Training Visualization
        tf.summary.scalar("Training Loss", self.loss)
        tf.summary.scalar("Training Accuracy", self.accuracy)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logdir, graph=tf.get_default_graph())
        self.counter = 0
        saver = tf.train.Saver()

        #Variables Initial
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        #Train
        model_path = self.model_dir + '/model%d'
        for iter in xrange(1, self.num_epoch + 1):
            print 'Epoch %d\n' % iter
            loss, accuracy = self.epoch(sess, images, labels)
            print '\tResult: Loss: %.5f Accuracy: %.2f' % (loss, accuracy * 100) + '%'
            saver.save(sess, model_path % iter)




