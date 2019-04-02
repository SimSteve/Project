import tensorflow as tf
import numpy as np
import src.CW.Carlini_Models as m
# import src.encryptions.unencrypted as e
import src.encryptions.permutated as e
from src.CW.l2_attack import CarliniL2
import time
import matplotlib.pyplot as plt

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
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

    plt.xlabel("{:2.0f}% it's {} (true label: {})".format(100 * np.max(prob), class_names[predicted_label],
                                                          class_names[true_label]), color=color)

    plt.draw()


def show(img):
    """
    Show MNSIT digits in the console.
    """
    img = img.reshape(28, 28)
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


'''
def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)
'''


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 1000
    num_of_examples = 1000

    x_test = x_test[:num_of_examples]
    y_test = y_test[:num_of_examples]

    batch_size = 50

    x_test = (x_test / 255.0) - 0.5
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape

    model = models["CW_1"](input_shape, encrypt=e.encrypt)
    model.load("mnist_CW_1_PERMUTATED")
    class_names = mnist_classes

    attack = CarliniL2(sess=sess, model=model, targeted=False, batch_size=batch_size)

    images = np.array(x_test)
    targets = np.eye(10)[np.array([y_test]).reshape(-1)]

    timestart = time.time()
    adv = attack.attack(images, targets)
    timeend = time.time()

    print("Took", timeend - timestart, "seconds to run", 1, "samples.")

    good = 0.0
    bad = 0.0

    g = open("safe_permutated", 'w')

    for i in range(len(adv)):
        real = y_test[i]
        prob_adv = model.model.predict(adv[i:i + 1])[0]  # the output is 2D array
        # prob_orig = model.model.predict(images[i:i + 1])[0]     # likewise

        prob_adv = softmax(prob_adv)
        # prob_orig = softmax(prob_orig)

        pred_adv = np.argmax(prob_adv).tolist()
        # pred_orig = np.argmax(prob_orig).tolist()

        # print("Real Classification: ", real)
        # print("Classification: ", pred_orig)
        # print("Adversarial Classification: ", pred_adv)
        # print("Total distortion:", np.sum((adv[i] - images[i]) ** 2) ** .5)

        # r.write("Real Classification: {}\n".format(real))
        # r.write("Classification: {}\n".format(pred_orig))
        # r.write("Adversarial Classification: {}\n".format(pred_adv))
        # r.write("Total distortion: {}\n\n".format(np.sum((adv[i] - images[i]) ** 2) ** .5))

        # plot_image(images[i], predicted_label=pred_orig, true_label=real, prob=prob_orig)

        # plot_image(adv[i], predicted_label=pred_adv, true_label=real, prob=prob_adv)

        # adv_img = np.reshape(adv[i], (28,28))
        # np.save("src/adversarial_images/adv1", adv_img)

        # plt.show()

        good += pred_adv == real
        bad += pred_adv != real

        if pred_adv != real:
            g.write("{}\n".format(i))

    g.close()

    test_acc = good / (good + bad)
    r = open("attcked_results_permutated", 'w')
    r.write("mnist_CW_1_PERMUTATED\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc,
                                                                                     (1.0 - test_acc) * 100))
    r.write("#####################################################\n")
    r.close()
