import tensorflow as tf
import numpy as np
import src.FGSM.Models as m
import time
import matplotlib.pyplot as plt
from src.FGSM.cleverhans.cleverhans.utils_keras import KerasModelWrapper
from src.FGSM.cleverhans.cleverhans.attacks.fast_gradient_method import FastGradientMethod

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _, (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    num_of_examples = 1000

    x_test = x_test[:num_of_examples]
    y_test = y_test[:num_of_examples]

    print(x_test.shape)

    x_test = x_test / 255.0
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape

    # import src.encryptions.permutated as e
    # name = "mnist_FGSM_PERMUTATED"

    # import src.encryptions.unencrypted as e
    # name = "mnist_FGSM_UNENCRYPTED"

    # import src.encryptions.permutated as e
    # name = "fashion_mnist_FGSM_PERMUTATED"

    import src.encryptions.unencrypted as e
    name = "fashion_mnist_FGSM_UNENCRYPTED"

    model = m.FGSM(input_shape, encrypt=e.encrypt)
    model.load(name)
    class_names = fashion_mnist_classes

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    adv_x = fgsm.generate_np(x_test, **fgsm_params)

    _, test_acc = model.evaluate(adv_x, y_test)
    print("unencrypted accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))

    # import src.encryptions.permutated as e
    # name = "mnist_FGSM_PERMUTATED"

    import src.encryptions.permutated as e
    name = "fashion_mnist_FGSM_PERMUTATED"

    model = m.FGSM(input_shape, encrypt=e.encrypt)
    model.load(name)

    _, test_acc = model.evaluate(adv_x, y_test)
    print("permutated accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))

    good = 0.0
    bad = 0.0

    safe = open("safe_indexes_FGSM_fashion", 'w')
    unsafe = open("unsafe_indexes_FGSM_fashion", 'w')

    for i in range(len(adv_x)):

        real = y_test[i]
        # enc_adv = np.reshape(e.encrypt(adv[i]), (1,28,28,1))
        prob_adv = sess.run(model.predict(np.float32(np.reshape(adv_x[i], (1, 28, 28, 1))))[0])  # the output is 2D array
        pred_adv = np.argmax(prob_adv).tolist()

        # enc_orig = np.reshape(e.encrypt(x_test[i]), (1, 28, 28, 1))
        prob_orig = sess.run(model.predict(np.float32(np.reshape(x_test[i], (1, 28, 28, 1))))[0])  # the output is 2D array
        pred_orig = np.argmax(prob_orig).tolist()

        good += pred_adv == real
        bad += pred_adv != real

        if pred_adv == real:
            safe.write("{}\n".format(i))
        else:
            if pred_orig == real:
                unsafe.write("{}*\n".format(i))  # images that the attacker successfully misleaded the model
            else:
                unsafe.write("{}\n".format(i))  # the model is wrong

        # if pred_adv != real and pred_orig == real:
        #     print("{}*\n".format(i))
    safe.close()
    unsafe.close()

    test_acc = good / (good + bad)
    print("accuracy: {:.2f}%\terror rate: {:.2f}%\n".format(100 * test_acc, (1.0 - test_acc) * 100))
