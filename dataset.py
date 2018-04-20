import numpy as np
import pickle

FLOAT_TYPE = np.float32
def preprocess_image(images, labels, num_class=10):
    length = len(images)
    dim = 32 * 32 * 3

    images = images.reshape((length, dim))
    images = zca_whiten(images)
    images = images.reshape((length, 32, 32, 3))

    new_labels = np.zeros((length, num_class), dtype=FLOAT_TYPE)
    new_labels[np.arange(length), labels[:, 0]] = 1

    return images, new_labels

def zca_whiten(X):
    EPS = 10e-5
    assert(X.ndim == 2)
    cov = np.dot(X.T, X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(1. / np.sqrt(d + EPS))
    W = np.dot(np.dot(E, D), E.T)
    X_white = np.dot(X, W)
    return X_white

def parse_data_cifar10():
    data = []
    labels = []
    for i in xrange(1, 6):
        dict = pickle.load(open('data/data_batch_%d' % i, 'rb'))
        length = len(dict['labels'])
        arr = np.array(dict['data'], dtype=FLOAT_TYPE)
        lab = np.array(dict['labels'], dtype=np.int).reshape((length, 1))
        data.append(arr)
        labels.append(lab)

    data = np.vstack(data)
    data = data.reshape((-1, 32, 32, 3))
    labels = np.vstack(labels)
    return data, labels

