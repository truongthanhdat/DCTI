import model
from dataset import *
import tensorflow as tf
import time
import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser('Testing DCTI')
    parser.add_argument('--dataset', help='Choose dataset: cifar10 (default), cifar100', type=str, default='cifar10')
    parser.add_argument('--num_epoch', help='Number of epoch (default = 100)',type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size (default = 128)', type=int, default=128)
    return parser.parse_args()

def load_data(num_class):
    path_test_data = 'data/cifar%d.pickle' % num_class

    if os.path.exists(path_test_data):
        print '%s found!' % path_test_data
        duration = time.time()
        dict = pickle.load(open(path_test_data, 'rb'))
        duration =  time.time() - duration
        print 'Loading Test Data takes %f second' % duration
        return dict['images'], dict['labels']

    print '%s not found. Preparing Dataset ...' % path_test_data
    duration = time.time()
    if num_class == 10:
        images, labels = parse_data_cifar10()
    else:
        images, labels = parse_data_cifar100()

    _, _, mean, std, W = preprocess_image(images, labels, num_class=num_class, mean=None, std=None, W=None)

    if num_class == 10:
        images, labels = parse_test_cifar10()
    else:
        images, labels = parse_test_cifar100()

    images, labels, _, _, _ = preprocess_image(images, labels, num_class=num_class, mean=mean, std=std, W=W)
    duration = time.time() - duration
    print 'Preparing Dataset takes %f second' % duration
    pickle.dump({'images': images, 'labels': labels}, open(path_test_data, 'wb'))
    return images, labels

if __name__ == '__main__':
    options = parse_args()

    if options.dataset == 'cifar10':
        NUM_CLASS = 10
    elif options.dataset == 'cifar100':
        NUM_CLASS = 100
    else:
        print 'Unknown Dataset: %s' % options.dataset
        exit(0)

    images, labels = load_data(NUM_CLASS)
    net = model.DCTI(num_class=NUM_CLASS, batch_size=options.batch_size, is_training=True)

    model_path = 'model/%s/model' % options.dataset
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    net.evaluate(sess, images, labels)

