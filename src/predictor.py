import matplotlib.pyplot as plt
import src.Models as mdl
import numpy as np
import tensorflow as tf

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []

class_types = {'fashion_mnist':fashion_mnist_classes, 'mnist':mnist_classes, 'cifar10':cifar10_classes}
data_types = {'fashion_mnist':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
models = {"CW_1": mdl.CW_1(), "CW_2": mdl.CW_2(), "FGSM": mdl.FGSM()}


def plot_image(predictions, true_label, img):
    '''
    Draws the image. ONLY for greyscale images (the third dimension should be 1)
    :param predictions: vector of probablities
    :param true_label: the true label
    :param img: the image itself
    :return:
    '''
    # in order to plot the image, we need a 2D array
    if len(np.array(img).shape) == 3:
        img = img[:, :, 0]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions),
                                         class_names[true_label]), color=color)
    plt.show()


def predict(model, data, labels, i=None):
    if i is None:
        # predicting the whole data
        predictions = model.predict(data)

        '''
        Do whatever you want with the predictions (plotting, calculating accuracies, ...)
        '''
    else:
        # adding a first dimension, batch-size
        img = np.expand_dims(data[i], axis=0)
        predictions = model.predict(img)

        plot_image(predictions, labels[i], data[i])


def main():
    global class_names, DATASET, MODEL

    data = data_types[DATASET]
    class_names = class_types[DATASET]

    # getting the desired model
    model = models[MODEL]

    # loading all the weights
    model.load(DATASET + "_" + MODEL + "_model")

    _, (x_test, y_test) = data.load_data()
    x_test = x_test / 255.0

    dims = np.array(x_test).shape

    changed_dims = False
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)
        changed_dims = True

    predict(model, x_test, y_test, i=1)


if __name__ == '__main__':
    DATASET = "mnist"
    MODEL = "CW_1"

    main()
