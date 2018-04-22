import model
from dataset import *
import tensorflow as tf
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Testing DCTI')
    parser.add_argument('--dataset', help='Choose dataset: cifar10 (default), cifar100', type=str, default='cifar10')
    parser.add_argument('--num_epoch', help='Number of epoch (default = 100)',type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size (default = 128)', type=int, default=128)
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_args()

    if options.dataset == 'cifar10':
        NUM_CLASS = 10
    elif options.dataset == 'cifar100':
        NUM_CLASS = 100
    else:
        print 'Unknown Dataset: %s' % options.dataset
        exit(0)


    net = model.DCTI(num_class=NUM_CLASS, batch_size=options.batch_size, is_training=False)

    print 'Preparing Dataset'
    duration = time.time()
    if NUM_CLASS == 10:
        images, labels = parse_data_cifar10()
    else:
        images, labels = parse_data_cifar100()

    _, _, mean, std, W = preprocess_image(images, labels, num_class=NUM_CLASS, mean=None, std=None, W=None)

    if NUM_CLASS == 10:
        images, labels = parse_test_cifar10()
    else:
        images, labels = parse_test_cifar100()

    images, labels, _, _, _ = preprocess_image(images, labels, num_class=NUM_CLASS, mean=mean, std=std, W=W)
    duration = time.time() - duration
    print 'Preparing Dataset takes %f second' % duration
    model_path = 'model/%s/model' % options.dataset
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    net.evaluate(sess, images, labels)

