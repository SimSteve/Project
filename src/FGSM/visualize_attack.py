import tensorflow as tf
import numpy as np
import src.FGSM.Models as m
import time
import matplotlib.pyplot as plt
from src.FGSM.cleverhans.cleverhans.attacks.fast_gradient_method import FastGradientMethod
from src.FGSM.cleverhans.cleverhans.utils_keras import KerasModelWrapper

fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_names = []


def plot_original_adversarial(orig_img, adv_img, orig_pred, adv_pred, orig_prob, adv_prob, true_label, distortion):
    if len(np.array(orig_img).shape) > 2:
        orig_img = orig_img[:, :, 0]

    if len(np.array(adv_img).shape) > 2:
        adv_img = adv_img[:, :, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle("Total distortion: {}".format(distortion))

    axes = [ax1, ax2]
    images = [orig_img, adv_img]
    preds = [orig_pred, adv_pred]
    probs = [orig_prob, adv_prob]
    titles = ["Original", "Adversarial"]

    for i in range(2):
        #axes[i].imshow(images[i], cmap=plt.cm.binary)  # row=0, col=0
        axes[i].imshow(images[i])
        axes[i].grid(False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].title.set_text(titles[i])

        if preds[i] == true_label:
            color = 'green'
        else:
            color = 'red'

        axes[i].set_xlabel("{:2.0f}% it's {} (true label: {})".format(100 * np.max(probs[i]), class_names[preds[i]],
                                                              class_names[true_label]), color=color)

    plt.show()


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    index = 3

    x_test = x_test[index:index+1]
    y_test = y_test[index:index+1]

    x_test = x_test / 255.0
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape

    import src.encryptions.permutated as e
    name = "mnist_FGSM_PERMUTATED"

    # import src.encryptions.unencrypted as e
    # name = "mnist_FGSM_UNENCRYPTED"

    model = m.FGSM(input_shape, encrypt=e.numpy_encrypt)
    model.load(name)
    class_names = mnist_classes

    images = np.array(x_test)
    targets = np.eye(10)[np.array(y_test).reshape(-1)]

    timestart = time.time()
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv = fgsm.generate_np(x_test, **fgsm_params)
    timeend = time.time()

    # model = m.FGSM_no_softmax(input_shape, encrypt=e.numpy_encrypt)
    # model.load(name)

    # plotting the images
    real = y_test[0]

    # enc_img_adv = e.numpy_encrypt(adv[0])
    # enc_img_orig = e.numpy_encrypt(images[0])
    #
    # prob_adv = model.model.predict(np.reshape(enc_img_adv, (1,28,28,1)))[0]  # the output is 2D array
    # prob_orig = model.model.predict(np.reshape(enc_img_orig, (1,28,28,1)))[0]  # likewise
    #
    # prob_adv = softmax(prob_adv)
    # prob_orig = softmax(prob_orig)

    prob_adv = sess.run(model.predict(np.float32(adv)))[0]
    prob_orig = sess.run(model.predict(np.float32(images)))[0]

    pred_adv = np.argmax(prob_adv)
    pred_orig = np.argmax(prob_orig)

    distortion = np.sum((adv[0] - images[0]) ** 2) ** .5

    print(pred_orig)
    print(pred_adv)
    print(real)
    print(prob_orig)
    print(prob_adv)

    plot_original_adversarial(images[0], adv[0], pred_orig, pred_adv, prob_orig, prob_adv, real, distortion)

    # adv_img = np.reshape(adv[i], (28,28))
    # np.save("src/adversarial_images/adv1", adv_img)