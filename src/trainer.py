import src.Models as mdl
import numpy as np
import tensorflow as tf

data_types = {'fashion_mnist':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
file_types = {'fashion_mnist':'fashion_mnist_FGSM_model', 'mnist':'mnist_FGSM_model', 'cifar10':'cifar10_FGSM_model'}

DATASET = "cifar10"
MODEL = "cw"


def main():
    global DATASET, MODEL

    m_file = file_types[DATASET]
    data = data_types[DATASET]

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if DATASET != 'cifar10':
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        input_shape = (28, 28, 1)
    else:
        input_shape = (32, 32, 3)

    # choosing the right model
    if MODEL == "fgsm":
        model = mdl.FGSM(m_file)
    elif MODEL == "cw":
        if DATASET == "cifar10":
            model = mdl.CW_cifar(m_file)
        else:
            model = mdl.CW_mnist(m_file)

    # building the networks' structure
    model.build(input_shape)

    # training
    model.train(x_train, y_train, ep=5)

    # evaluating
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

    # saving model
    model.save()


if __name__ == '__main__':
    main()
