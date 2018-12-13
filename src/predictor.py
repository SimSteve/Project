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
file_types = {'fashion_mnist':'fashion_mnist_FGSM_model', 'mnist':'mnist_FGSM_model', 'cifar10':'cifar10_FGSM_model'}

DATASET = "mnist"
MODEL = "cw"


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]), color=color)
    plt.show()


def main(type, model_type):
    global class_names, DATASET, MODEL

    m_file = file_types[type]
    data = data_types[type]
    class_names = class_types[type]

    # loading a saved model
    if MODEL == "fgsm":
        model = mdl.FGSM(m_file).load()
    elif MODEL == "cw":
        if DATASET == "cifar10":
            model = mdl.CW_cifar(m_file).load()
        else:
            model = mdl.CW_mnist(m_file).load()

    _, (x_test, y_test) = data.load_data()
    x_test = x_test / 255.0

    # expanding the images to get a third dimension (needed for conv layers)
    if type != 'cifar10':
        x_test = np.expand_dims(x_test, -1)

    # predicting the whole data
    predictions = model.predict(x_test)

    # resizing the images to their original dimensions
    if type != 'cifar10':
        x_test = x_test[:, :, :, 0]

    plt.figure(figsize=(6, 3))
    plot_image(i=45, predictions_array=predictions, true_label=y_test, img=x_test)


if __name__ == '__main__':
    main("mnist", "cw")
