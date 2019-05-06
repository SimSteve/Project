import tensorflow as tf
import numpy as np
import src.Models as m
import time
import matplotlib.pyplot as plt
from src.FGSM.cleverhans.cleverhans.utils_keras import KerasModelWrapper
from src.FGSM.cleverhans.cleverhans.attacks.fast_gradient_method import FastGradientMethod

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []


def filter(inputs, model, labels):
    f_x = []
    f_y = []

    pred = [model.predict(inputs[i:i+1]) for i in range(len(labels))]
    pred = [np.argmax(pr) for pr in pred]

    for i in range(len(inputs)):
        if pred[i] == labels[i]:
            f_x.append(inputs[i])
            f_y.append(labels[i])
    return np.array(f_x), f_y


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    num_of_examples = 1000

    x_test = x_test[:num_of_examples]
    y_test = y_test[:num_of_examples]

    x_test = x_test / 255.0
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape

    # import src.encryptions.permutated as e
    # name = "mnist_FGSM_PERMUTATED"

    import src.encryptions.unencrypted as e
    name = "mnist_FGSM_UNENCRYPTED"

    # import src.encryptions.permutated as e
    # name = "fashion_mnist_FGSM_PERMUTATED"

    # import src.encryptions.unencrypted as e
    # name = "fashion_mnist_FGSM_UNENCRYPTED"

    model = m.FGSM(input_shape, encrypt=e.encrypt)
    model.load(name)
    class_names = fashion_mnist_classes

    x_test, y_test = filter(x_test, model, y_test)
    print(len(x_test))

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_x = fgsm.generate_np(x_test, **fgsm_params)

    import src.encryptions.permutated as e
    name = "mnist_FGSM_PERMUTATED"

    # import src.encryptions.permutated as e
    # name = "fashion_mnist_FGSM_PERMUTATED"

    model = m.FGSM(input_shape, encrypt=e.numpy_encrypt)
    model.load(name)

    adv_pred_prob = model.predict(adv_x)
    orig_pred_prob = model.predict(x_test)

    adv_pred = [np.argmax(pr) for pr in adv_pred_prob]
    orig_pred = [np.argmax(pr) for pr in orig_pred_prob]

    adv_acc = np.mean(np.equal(adv_pred, y_test))
    orig_acc = np.mean(np.equal(orig_pred, y_test))

    print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * adv_acc, (1.0 - adv_acc) * 100))

    # r = open("attacked_results", 'a')
    # r.write("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(name, 100 * adv_acc,
    #                                                               (1.0 - adv_acc) * 100))
    # r.write("#####################################################\n")
    # r.close()
    #
    # print("{}\taccuracy: {:.2f}%\terror rate: {:.2f}%\n".format(name, 100 * adv_acc,
    #                                                               (1.0 - adv_acc) * 100))
