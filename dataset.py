import numpy as np
import pickle
import cPickle

FLOAT_TYPE = np.float32
def preprocess_image(images, labels, num_class=10, mean=None, std=None, W=None):
    length = len(images)

    if mean is None:
        mean = images.mean(axis=0)

    if std is None:
        std = images.std(axis=0)

    images = (images - mean) / std

    if W is None:
        images, W = zca_whiten(images)
    else:
        images = np.dot(images, W)

    images = images.reshape((length, 3, 32, 32)).transpose(0, 2, 3, 1)
    new_labels = np.zeros((length, num_class), dtype=FLOAT_TYPE)
    new_labels[np.arange(length), labels[:, 0]] = 1

    return images, new_labels, mean, std, W

def zca_whiten(X):
    EPS = 10e-5
    assert(X.ndim == 2)
    cov = np.dot(X.T, X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(1. / np.sqrt(d + EPS))
    W = np.dot(np.dot(E, D), E.T)
    X_white = np.dot(X, W)
    return X_white, W

def parse_data_cifar10():
    data = []
    labels = []
    for i in xrange(1, 6):
        dict = pickle.load(open('data/cifar10/data_batch_%d' % i, 'rb'))
        length = len(dict['labels'])
        arr = dict['data'].astype(FLOAT_TYPE)
        lab = np.array(dict['labels'], dtype=np.int).reshape((length, 1))
        data.append(arr)
        labels.append(lab)

    data = np.vstack(data)
    labels = np.vstack(labels)
    return data, labels

def parse_test_cifar10():
    dict = pickle.load(open('data/cifar10/test_batch', 'rb'))
    data = dict['data'].astype(FLOAT_TYPE)
    length = len(data)
    labels = np.array(dict['labels'], dtype=np.int).reshape((length, 1))
    return data, labels

def parse_data_cifar100():
    file = cPickle.load(open('data/cifar100/train', 'rb'))
    data = file['data'].astype(FLOAT_TYPE)
    length = len(data)
    labels = np.array(file['fine_labels'], dtype=np.int).reshape(length, 1)
    return data, labels

def parse_test_cifar100():
    dict = pickle.load(open('data/cifar100/test', 'rb'))
    data = dict['data'].astype(FLOAT_TYPE)
    length = len(data)
    labels = np.array(dict['fine_labels'], dtype=np.int).reshape((length, 1))
    return data, labels


