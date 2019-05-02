import tensorflow as tf
import numpy as np
import src.CW.Models as m
from src.CW.l2_attack import CarliniL2
from src.CW.l0_attack import CarliniL0
from src.CW.li_attack import CarliniLi
import time
import matplotlib.pyplot as plt

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []

models = {"CW_1": m.CW_1, "CW_2": m.CW_2}


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def failed(adv):
    return (adv.flatten() == [0.0 for _ in range(28*28)]).all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    num_of_examples = 1000

    x_test = x_test[:num_of_examples]
    y_test = y_test[:num_of_examples]

    batch_size = 1

    x_test = (x_test / 255.0) - 0.5
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape

    # import src.encryptions.permutated as e
    # name = "mnist_CW_1_PERMUTATED_0.5NORM"

    import src.encryptions.unencrypted as e
    name = "mnist_CW_1_UNENCRYPTED_0.5NORM"

    # import src.encryptions.permutated as e
    # name = "fashion_mnist_CW_1_PERMUTATED_0.5NORM"

    # import src.encryptions.unencrypted as e
    # name = "fashion_mnist_CW_1_UNENCRYPTED_0.5NORM"

    model = models["CW_1"](input_shape, encrypt=e.encrypt)
    model.load(name)
    class_names = mnist_classes

    # attack = CarliniL2(sess=sess, model=model, targeted=False, max_iterations=1000)
    attack = CarliniL0(sess=sess, model=model, targeted=False, max_iterations=1000)
    # attack = CarliniLi(sess=sess, model=model, targeted=False, max_iterations=1000)

    images = np.array(x_test)
    targets = np.eye(10)[np.array(y_test).reshape(-1)]

    timestart = time.time()
    adv = attack.attack(images, targets)
    timeend = time.time()

    print("Took", timeend - timestart, "seconds to run", 1, "samples.")

    import src.encryptions.permutated as e
    name = "mnist_CW_1_PERMUTATED_0.5NORM"

    # import src.encryptions.permutated as e
    # name = "fashion_mnist_CW_1_PERMUTATED_0.5NORM"

    model = models["CW_1"](input_shape, encrypt=e.numpy_encrypt)
    model.load(name)

    good = 0.0
    bad = 0.0

    safe = open("safe_indexes_l0_mnist", 'w')
    unsafe = open("unsafe_indexes_l0_mnist", 'w')

    for i in range(len(adv)):
        if failed(adv[i]):
            safe.write("{}\n".format(i))
            good += 1
            continue

        real = y_test[i]
        # enc_adv = np.reshape(e.encrypt(adv[i]), (1,28,28,1))
        prob_adv = sess.run(model.predict(np.float32(np.reshape(adv[i], (1,28,28,1))))[0])          # the output is 2D array
        prob_adv = softmax(prob_adv)
        pred_adv = np.argmax(prob_adv).tolist()

        # enc_orig = np.reshape(e.encrypt(x_test[i]), (1, 28, 28, 1))
        prob_orig = sess.run(model.predict(np.float32(np.reshape(x_test[i], (1, 28, 28, 1))))[0])   # the output is 2D array
        prob_orig = softmax(prob_orig)
        pred_orig = np.argmax(prob_orig).tolist()

        good += pred_adv == real
        bad += pred_adv != real

        if pred_adv == real:
            safe.write("{}\n".format(i))
        else:
            if pred_orig == real:
                unsafe.write("{}*\n".format(i))     # images that the attacker successfully misleaded the model
            else:
                unsafe.write("{}\n".format(i))      # the model is wrong

        # if pred_adv != real and pred_orig == real:
        #     print("{}*\n".format(i))
    safe.close()
    unsafe.close()

    test_acc = good / (good + bad)
    print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))

    # r = open("attacked_results", 'a')
    # r.write("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(name, 100 * test_acc,
    #                                                                                  (1.0 - test_acc) * 100))
    # r.write("#####################################################\n")
    # r.close()
