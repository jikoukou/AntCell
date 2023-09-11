from Algorithm import Algorithm
import keras
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def main():
    maxBlocks = 5
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize and reshape data

    x_train, x_test = x_train.astype(
        float) / 255.0, x_test.astype(float) / 255.0

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    algorithm = Algorithm(maxBlocks, x_train, y_train,
                          x_test, y_test, max_attempts=5)
    mnist_model = algorithm.generate_model(
        is_cifar=False, search_epochs=5, final_epochs=5)


if __name__ == "__main__":
    main()
