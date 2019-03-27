import tensorflow as tf
import numpy as np
import src.CW.Carlini_Models as m
import src.encryptions.unencrypted as e
#import src.encryptions.permutated as e
from src.CW.l2_attack import CarliniL2
import time
import matplotlib.pyplot as plt

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []

models = {"CW_1": m.CW_1, "CW_2": m.CW_2}

def plot_image(img, true_label, predicted_label, prob):
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

    _, ax = plt.subplots()
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{:2.0f}% it's {} (true label: {})".format(100 * np.max(prob), class_names[predicted_label], class_names[true_label]), color=color)

    plt.draw()


def show(img):
    """
    Show MNSIT digits in the console.
    """
    img = img.reshape(28,28)
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_test = (x_test / 255.0) - 0.5
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape
    model = models["CW_1"](input_shape, encrypt=e.encrypt)
    model.load("mnist_CW_1_UNENCRYPTED")
    class_names = mnist_classes
    #model.load("mnist_CW_1_PERMUTATED")

    attack = CarliniL2(sess=sess, model=model, targeted=False)

    images = np.array([x_test[1]])
    true_label = [y_test[1]]
    targets = np.array([[0,0,1,0,0,0,0,0,0,0]])     # TODO change to real label

    timestart = time.time()
    adv = attack.attack(images, targets)
    timeend = time.time()

    print("Took", timeend - timestart, "seconds to run", 1, "samples.")

    for i in range(len(adv)):
        real = true_label[i]
        prob_adv = model.model.predict(adv[i:i + 1])[0]         # the output is 2D array
        prob_orig = model.model.predict(images[i:i + 1])[0]     # likewise

        prob_adv = softmax(prob_adv)
        prob_orig = softmax(prob_orig)

        pred_adv = np.argmax(prob_adv).tolist()
        pred_orig = np.argmax(prob_orig).tolist()

        print("Real Classification: ", real)
        print("Classification: ", pred_orig)
        print("Adversarial Classification: ", pred_adv)
        print("Total distortion:", np.sum((adv[i] - images[i]) ** 2) ** .5)

        plot_image(images[i], predicted_label=pred_orig, true_label=real, prob=prob_orig)

        plot_image(adv[i], predicted_label=pred_adv, true_label=real, prob=prob_adv)

        plt.show()

