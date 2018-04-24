import model
import dataset
import time
import tensorflow as tf
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Training DCTI')
    parser.add_argument('--dataset', help='Choose dataset: cifar10 (default), cifar100', type=str, default='cifar10')
    parser.add_argument('--num_epoch', help='Number of epoch (default = 100)',type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size (default = 128)', type=int, default=128)
    parser.add_argument('--seed', help='seed value (default=None)', type=int, default=None)
    parser.add_argument('--learning_rate', help='Learning rate (default = 0.001)', type=float, default=0.001)
    parser.add_argument('--regularization_rate', help='Regularization rate (default=0.0005)', type=float, default=0.0005)
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

    print 'Prepareing Data'
    duration = time.time()
    if NUM_CLASS == 10:
         images, labels = dataset.parse_data_cifar10()
    else:
         images, labels = dataset.parse_data_cifar100()
    images, labels, _, _, _ = dataset.preprocess_image(images, labels, num_class=10)
    duration = time.time() - duration
    print 'Prepareing Data take %f second' % duration

    model_dir = 'model/' + options.dataset
    logdir = 'logs/' + options.dataset

    sess = tf.Session()

    net = model.DCTI(batch_size=options.batch_size, num_epoch=options.num_epoch, seed=options.seed, num_class=NUM_CLASS, learning_rate=options.learning_rate, \
            regularization_rate=options.regularization_rate, model_dir=model_dir, logdir=logdir)

    net.train(sess, images, labels)
