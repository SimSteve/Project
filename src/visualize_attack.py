import importlib
from pathlib import PurePath
import src.padding as p
import tensorflow as tf
import numpy as np
import src.Models as m
import matplotlib.pyplot as plt
import sys

data_types = {'fashion':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                         'Ankle boot']
mnist_classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_types = {'fashion':fashion_mnist_classes, 'mnist':mnist_classes, 'cifar10':cifar10_classes}
class_names = []
models = {"modelA": m.modelA, "modelB": m.modelB}

train_mode = {"CTR": "ctr", "CBC": "cbc", "ECB": "ecb",
              "PERMUTATED": "permutated", "UNENCRYPTED": "unencrypted"}


DATASET = "DATASET"
MODEL = "MODEL"
TRAIN_WITH_ME = "TRAIN_WITH_ME"
PADDING = "PADDING"
NORM = "NORM"
FILE = "FILE"
INDEX = "INDEX"
CARLINI = "CARLINI"

params = {DATASET: None, TRAIN_WITH_ME: None, PADDING: 0, NORM: 0, FILE:None, CARLINI:None}


def plot_original_adversarial(orig_img, adv_img, orig_pred, adv_pred, orig_prob, adv_prob, true_label):
    if len(np.array(orig_img).shape) > 2:
        orig_img = orig_img[:, :, 0]

    if len(np.array(adv_img).shape) > 2:
        adv_img = adv_img[:, :, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)

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


# getting command line arguments
if len(sys.argv) == 1:
    print("Missing arguments. Try -h for help")
    exit()

if sys.argv[1] == '-h':
    print("Welcome to our predicting tool:")
    print("\t-f\tspecifying the filename of the model (mandatory)")
    print("\t-i\tspecifying the index (mandatory)")
    print("\t-c\tspecifying carlini mode; 2,0 or i. default is 2 (optional)")
    exit()

for i in range(1, len(sys.argv)):
    if sys.argv[i] == '-f':
        if ".h5" in sys.argv[i + 1]:
            params[FILE] = PurePath(sys.argv[i + 1]).parts[-1].split(".h5")[0]
        else:
            params[FILE] = sys.argv[i + 1]
    if sys.argv[i] == '-i':
        params[INDEX] = int(sys.argv[i + 1], 10)
    if sys.argv[i] == '-c':
        params[CARLINI] = sys.argv[i+1]

MODEL_NAME = params[FILE]

name_parts = MODEL_NAME.split('_')
for i, part in enumerate(name_parts):
    if part == "modelA":
        params[MODEL] = part
        from src.CW import CW_attack as a
        a.set_mode(params[CARLINI])
    if part == "modelB":
        params[MODEL] = part
        from src.FGSM import FGSM_attack as a
    if part == "mnist" or part == "fashion":
        params[DATASET] = part
    if part == "UNENCRYPTED" or part == "PERMUTATED":
        params[TRAIN_WITH_ME] = part
    if "NORM" in part:
        params[NORM] = float(part.split("NORM")[0])
    if "PADDED" in part:
        params[PADDING] = int(part.split("PADDED")[0])

print("DATASET\t= {}".format(params[DATASET]))
print("MODEL\t= {}".format(params[MODEL]))
print("TRAINER\t= {}".format(params[TRAIN_WITH_ME]))
print("NORM\t= {}".format(params[NORM]))
print("PADDING\t= {}".format(params[PADDING]))
print("NAME\t= {}".format(MODEL_NAME))
print("INDEX\t= {}".format(params[INDEX]))

data = data_types[params[DATASET]]
class_names = class_types[params[DATASET]]

_, (x_test, y_test) = data.load_data()

img = x_test[params[INDEX]]
label = y_test[params[INDEX]]

img = img / 255.0 - params[NORM]
# TODO
# padding
# img = p.pad(img, number_of_paddings=params[PADDING], padder=0.0)
img = np.reshape(img, (1,28,28,1))

# white-box attack, so the attacker gets to know only the architecture of the model
# which is like giving him the unencrypted version
model = MODEL_NAME.replace("PERMUTATED", "UNENCRYPTED")

adv = a.attack(img, label, model)

helper = importlib.import_module("src.encryptions." + train_mode[params[TRAIN_WITH_ME]])
input_shape = np.array(img[0]).shape
model = models[params[MODEL]](input_shape, encrypt=helper.encrypt)
model.load(MODEL_NAME)

prob_adv = model.predict(np.float32(adv))[0]  # the output is 2D array
prob_orig = model.predict(np.float32(img))[0] # likewise

pred_adv = np.argmax(prob_adv)
pred_orig = np.argmax(prob_orig)

plot_original_adversarial(img[0], adv[0], pred_orig, pred_adv, prob_orig, prob_adv, label)

# adv_img = np.reshape(adv[i], (28,28))
# np.save("src/adversarial_images/adv1", adv_img)
