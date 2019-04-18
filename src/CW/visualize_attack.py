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

    index = 5555

    x_test = x_test[index:index+1]
    y_test = y_test[index:index+1]

    x_test = (x_test / 255.0) - 0.5
    dims = np.array(x_test).shape
    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_test[0]).shape

    import src.encryptions.permutated as e
    name = "mnist_CW_1_PERMUTATED_0.5NORM"

    # import src.encryptions.unencrypted as e
    # name = "fashion_mnist_CW_1_UNENCRYPTED_0.5NORM"

    model = models["CW_1"](input_shape, encrypt=e.encrypt)
    model.load(name)
    class_names = mnist_classes

    attack = CarliniL2(sess=sess, model=model, targeted=False, max_iterations=1000)
    # attack = CarliniL0(sess=sess, model=model, targeted=False, max_iterations=1000)
    # attack = CarliniLi(sess=sess, model=model, targeted=False, max_iterations=1000)

    images = np.array(x_test)
    targets = np.eye(10)[np.array(y_test).reshape(-1)]

    timestart = time.time()
    adv = attack.attack(images, targets)
    timeend = time.time()

    if (adv[0].flatten() == [0.0 for _ in range(28*28)]).all():
        print("failed")
        #exit()

    # plotting the images
    real = y_test[0]

    # enc_img_adv = e.encrypt(adv[0])
    # enc_img_orig = e.encrypt(images[0])

    prob_adv = sess.run(model.predict(np.float32(np.reshape(adv[0], (1,28,28,1))))[0])      # the output is 2D array
    prob_orig = sess.run(model.predict(np.float32(np.reshape(images[0], (1,28,28,1))))[0])  # likewise

    prob_adv = softmax(prob_adv)
    prob_orig = softmax(prob_orig)

    pred_adv = np.argmax(prob_adv).tolist()
    pred_orig = np.argmax(prob_orig).tolist()

    distortion = np.sum((adv[0] - images[0]) ** 2) ** .5

    plot_original_adversarial(images[0], adv[0], pred_orig, pred_adv, prob_orig, prob_adv, real, distortion)

    # adv_img = np.reshape(adv[i], (28,28))
    # np.save("src/adversarial_images/adv1", adv_img)
