from tensorflow.keras.utils import to_categorical
import pickle
import tensorflow as tf
import sys

dataset = 'mnist'
# old_stdout = sys.stdout
# sys.stdout = open('mnist.txt', 'w')


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def load_CIFAR_batch(filename):
    """ load single batch of cifar"""

    datadict = unpickle(filename)

    X = datadict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    X = X.astype('float')/255
    Y = datadict['labels']

    X = X.reshape(10000, 32, 32, 3)

    Y = to_categorical(Y)

#     Y = np.array(Y)
    return X, Y


def get_channel_axis():
    return 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
