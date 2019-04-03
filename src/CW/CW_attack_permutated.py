import tensorflow as tf
import numpy as np
import src.CW.Carlini_Models as m
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

'''
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
'''


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def get_target(x):
    return 9 - x


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 1.0

    # 1000
    num_of_examples = 100

    x_test = x_test[:num_of_examples]
    y_test = y_test[:num_of_examples]

    batch_size = 50

    x_test = (x_test / 255.0) - 0.5
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    for i,img in enumerate(x_test):
        x_test[i] = e.encrypt(img)

    input_shape = np.array(x_test[0]).shape

    attacked_name = "mnist_CW_1_PERMUTATED_0.5NORM_SEED=42"

    attacked_model = models["CW_1"](input_shape, encrypt=e.encrypt)
    attacked_model.load(attacked_name)

    safe_name = "mnist_CW_1_PERMUTATED_0.5NORM_SEED=79"

    safe_model = models["CW_1"](input_shape, encrypt=e.encrypt)
    safe_model.load(safe_name)

    class_names = mnist_classes

    attack = CarliniL2(sess=sess, model=attacked_model, targeted=False, batch_size=batch_size)

    images = np.array(x_test)

    targets = np.eye(10)[np.array([y_test]).reshape(-1)]        # targets are the true labels

    timestart = time.time()
    adv = attack.attack(images, targets)
    timeend = time.time()

    print("Took", timeend - timestart, "seconds to run", 1, "samples.")

    A_good = 0.0
    A_bad = 0.0
    S_good = 0.0
    S_bad = 0.0

    attacked_file = open("attacked_model_successfully_predicted_indexes", 'w')
    safe_file = open("safe_model_successfully_attacked_indexes", 'w')

    for i in range(len(adv)):
        real = y_test[i]

        e.seed = 42
        prob_attacked = attacked_model.model.predict(adv[i:i + 1])[0]  # the output is 2D array
        prob_attacked = softmax(prob_attacked)
        pred_attacked = np.argmax(prob_attacked).tolist()

        A_good += pred_attacked == real
        A_bad += pred_attacked != real

        if pred_attacked == real:
            attacked_file.write("{}\n".format(i))

        e.seed = 79
        prob_safe = safe_model.model.predict(adv[i:i + 1])[0]  # the output is 2D array
        prob_safe = softmax(prob_safe)
        pred_safe = np.argmax(prob_safe).tolist()

        S_good += pred_safe == real
        S_bad += pred_safe != real

        if pred_safe != real:
            safe_file.write("{}\n".format(i))

    attacked_file.close()
    safe_file.close()

    test_acc = A_good / (A_good + A_bad)
    r = open("attacked_results_permutated_non_targeted_100", 'w')
    r.write("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(attacked_name, 100 * test_acc,
                                                                                     (1.0 - test_acc) * 100))
    r.write("#####################################################\n")

    test_acc = S_good / (S_good + S_bad)
    r.write("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(safe_name, 100 * test_acc,
                                                                  (1.0 - test_acc) * 100))
    r.write("#####################################################\n")

    r.close()
