import numpy as np
import pickle

FLOAT_TYPE = np.float32
def preprocess_image(images, labels, num_class=10):
    length = len(images)
    images = images - 128
    images = images / 128
    new_labels = np.zeros((length, num_class), dtype=FLOAT_TYPE)
    new_labels[np.arange(length), labels[:, 0]] = 1
    return images, new_labels

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

