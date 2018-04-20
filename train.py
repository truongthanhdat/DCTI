import model
import dataset

import tensorflow as tf
if __name__ == '__main__':
    images, labels = dataset.parse_data_cifar10()
    images, labels = dataset.preprocess_image(images, labels, num_class=10)
    sess = tf.Session()
    net = model.DCTI(learning_rate=0.001, batch_size=256, regularization_rate=0.0005)
    net.train(sess, images, labels)
