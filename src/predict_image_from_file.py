import importlib

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
models = {"CW_1": mdl.CW_1, "CW_2": mdl.CW_2, "FGSM": mdl.FGSM}
train_mode = {"CTR":"ctr","CBC":"cbc","ECB":"ecb", "PERMUTATED":"permutated", "UNENCRYPTED":"unencrypted"}


def plot_image(image, prediction, prob):
    '''
    Draws the image. ONLY for greyscale images (the third dimension should be 1)
    :param predictions: vector of probablities
    :param true_label: the true label
    :param img: the image itself
    :return:
    '''
    # in order to plot the image, we need a 2D array
    image = np.reshape(image, (28,28))

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    plt.xlabel("{:2.0f}% it's {}".format(100 * np.max(prob), class_names[prediction]))
    plt.show()


img = np.load("src/adversarial_images/adv1.npy")
img = np.reshape(img, (1,28,28,1))
model = mdl.CW_1((28,28,1))
model.load("mnist_CW_1_UNENCRYPTED")
prob = model.predict(img)
pred = np.argmax(prob[0])
class_names = mnist_classes
plot_image(img, pred, prob)

