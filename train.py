import model
import dataset
import time
import tensorflow as tf

if __name__ == '__main__':
    print 'Prepareing Data'
    duration = time.time()
    images, labels = dataset.parse_data_cifar10()
    images, labels = dataset.preprocess_image(images, labels, num_class=10)
    duration = time.time() - duration
    print 'Prepareing Data take %f second' % duration

    sess = tf.Session()
    net = model.DCTI(learning_rate=0.001, batch_size=256, regularization_rate=0.0005, num_epoch=1000)
    net.train(sess, images, labels)
