import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys
import time
import os

FLOAT_TYPE = tf.float32

class DCTI:
    def __init__(self, num_class=10, learning_rate=0.001, regularization_rate=0.0005, batch_size=256, num_epoch=200, logdir='logs', model_dir='model', seed=1, is_training=True):
        #Setup general config
        self.num_class = num_class
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.is_training = is_training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 1000, 0.9, staircase=True)
        self.regularization_rate = regularization_rate
        self.logdir = logdir
        self.model_dir = model_dir

        #Setup Network
        self.images = tf.placeholder(shape=[None, 32, 32, 3], dtype=FLOAT_TYPE)
        self.labels = tf.placeholder(shape=[None, self.num_class], dtype=FLOAT_TYPE)
        with slim.arg_scope([slim.conv2d, slim.fully_connected], \
                weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, seed=seed), \
                trainable=self.is_training):
            self.net=  self.build_network(is_training=self.is_training)

        #Softmax loss
        self.softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.net, labels=self.labels))

        #Regularization loss
        self.weights = tf.trainable_variables()
        self.regularization_loss = tf.add_n([ tf.nn.l2_loss(weight) for weight in self.weights ]) * self.regularization_rate

        #Total Loss and Accuracy
        self.loss = self.softmax_loss + self.regularization_loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.net, axis=1), tf.argmax(self.labels, axis=1)), FLOAT_TYPE))

        #Setup optimize
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step)

        #Training Visualization
        tf.summary.scalar("Training Loss", self.loss)
        tf.summary.scalar("Training Accuracy", self.accuracy)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logdir, graph=tf.get_default_graph())
        self.counter = 0

    def build_network(self, is_training=True):
        # First Phase 32x32
        net = slim.conv2d(self.images, 64, [3, 3], scope='conv1_1')     #Conv2D: 3x3x64
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training) #Dropout: keep_prob = 0.7
        net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')             #Conv2D: 3x3x64
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool1')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training) #Dropout: keep_prob = 0.7

        # Second Phase 16x16
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')            #Conv2D: 3x3x128
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training) #Dropout: keep_prob = 0.7
        net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')            #Conv2D: 3x3x128
        net = slim.batch_norm(net)                                      #Barch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool2')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training) #Dropout: keep_prob = 0.7

        # Third Phase 8x8
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')            #Conv2D: 3x3x256
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.dropout(net, keep_prob=0.6, is_training=is_training) #Dropout: keep_prob = 0.6
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')            #Conv2D: 3x3x256
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.dropout(net, keep_prob=0.6, is_training=is_training) #Dropout: keep_prob = 0.6
        net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')            #Conv2D: 3x3x256
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool3')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.6, is_training=is_training) #Dropout: keep_prob = 0.6

        #Fourth Phase 4x4
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')            #Conv2D: 3x3x512
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.dropout(net, keep_prob=0.6)                          #Dropout: keep_prob = 0.6
        net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')            #Conv2D: 3x3x512
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.max_pool2d(net, [2, 2], scope='pool4')               #Max Pooling: 2x2
        net = slim.dropout(net, keep_prob=0.6, is_training=is_training) #Dropout: keep_prob = 0.6

        #Fifth Phase 2x2
        net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')            #Conv2D: 3x3x512
        net = slim.batch_norm(net)                                      #Batch Norm
        net = slim.avg_pool2d(net, [2, 2], scope='pool5')               #Global Average Pooling
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training) #Dropout: keep_prob = 0.5

        # Fully Connected
        net = slim.flatten(net, scope='flatten')                                            #Flatten
        net = slim.fully_connected(net, self.num_class, activation_fn=None, scope='fc')     #Fully Connected: 512 >> 10
        net = slim.batch_norm(net)                                                          #Batch Norm

        return net


    def backward(self, sess, images, labels):
        lr, _, loss, accuracy, summary = sess.run([self.optimizer._lr, self.trainer, self.loss, self.accuracy, self.summary_op], feed_dict={self.images: images, self.labels: labels})
        self.counter = self.counter + 1
        self.summary_writer.add_summary(summary, self.counter)
        return lr, loss, accuracy

    def epoch(self, sess, images, labels):
        total_loss = 0.0
        total_accuracy = 0.0
        first = 0
        length = len(images)
        iter = 0
        while (first < length):
            last = min(length, first + self.batch_size)
            duration = time.time()
            lr, loss, accuracy = self.backward(sess, images[first:last], labels[first:last])
            duration = time.time() - duration
            total_loss = total_loss + loss * (last - first)
            total_accuracy = total_accuracy + accuracy * (last - first)
            first = last
            iter = iter + 1
            sys.stdout.write("\033[F") # Cursor up one line
            print '\tIteration %d: Loss: %.5f Accuracy: %.2f' % (iter, loss, accuracy * 100) + '%' + ' Time: %f second Learning Rate %f' % (duration, lr)

        return total_loss / float(length), total_accuracy / float(length)

    def train(self, sess, images, labels):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        #Variables Initial

        #Train
        model_path = self.model_dir + '/model'
        for iter in xrange(1, self.num_epoch + 1):
            print 'Epoch %d\n' % iter
            duration = time.time()
            loss, accuracy = self.epoch(sess, images, labels)
            duration = time.time() - duration
            print '\tResult: Loss: %.5f Accuracy: %.2f' % (loss, accuracy * 100) + '%' + ' Time: %f second' % (duration)
            saver.save(sess, model_path)

    def evaluate(self, sess, images, labels):
        first = 0
        length = len(images)
        total_accuracy = 0.0
        iter = 0
        print 'Evaluate\n'
        while (first < length):
            last = min(length, first + self.batch_size)
            accuracy = sess.run(self.accuracy, feed_dict={self.images: images[first:last], self.labels: labels[first:last]})
            total_accuracy = total_accuracy + accuracy * (last - first)
            first = last
            sys.stdout.write("\033[F") # Cursor up one line
            print 'Accuracy on batch %d: %f' % (iter + 1, accuracy * 100) + '%'
            iter = iter + 1
        print 'Accuracy: %f' % (total_accuracy * 100 / float(length)) + '%'


