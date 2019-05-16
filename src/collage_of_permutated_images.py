import tensorflow as tf
import matplotlib.pyplot as plt
import src.padding as p
import numpy as np
import sys
import src.encryptions.permutated as e

data_types = {'fashion':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist}

DATASET = "DATASET"
PADDING = "PADDING"
NUM_OF_CLASSES = "NUM_OF_CLASSES"
NUM_OF_EXAMPLES_PER_CLASS ="NUM_OF_EXAMPLES_PER_CLASS"


params = {DATASET: None, PADDING: 0, NUM_OF_CLASSES:10, NUM_OF_EXAMPLES_PER_CLASS:10}

# getting command line arguments
if len(sys.argv) == 1:
    print("Missing arguments. Try -h for help")
    exit()

if sys.argv[1] == '-h':
    print("Welcome to our collage tool:")
    print("\t-d\tspecifying the dataset; mnist or fashion (mandatory)")
    print("\t-p\tspecifying the number of rows to pad, default is 0 (optional)")
    print("\t-c\tspecifying the number of classes, default is 10 (optional)")
    print("\t-i\tspecifying the number images for each class, default is 10 (optional)")
    exit()

for i in range(1, len(sys.argv)):
    if sys.argv[i] == '-d':
        params[DATASET] = sys.argv[i + 1]
    if sys.argv[i] == '-p':
        params[PADDING] = int(sys.argv[i + 1], 10)
    if sys.argv[i] == '-c':
        params[NUM_OF_CLASSES] = int(sys.argv[i + 1], 10)
    if sys.argv[i] == '-i':
        params[NUM_OF_EXAMPLES_PER_CLASS] = int(sys.argv[i + 1], 10)


print("DATASET\t\t\t\t\t\t= {}".format(params[DATASET]))
print("PADDING\t\t\t\t\t\t= {}".format(params[PADDING]))
print("NUM_OF_CLASSES\t\t\t\t= {}".format(params[NUM_OF_CLASSES]))
print("NUM_OF_EXAMPLES_PER_CLASS\t= {}".format(params[NUM_OF_EXAMPLES_PER_CLASS]))

data = data_types[params[DATASET]]

(x_train, y_train), _ = data.load_data()

num_of_classes = params[NUM_OF_CLASSES]
num_of_examples_per_class = params[NUM_OF_EXAMPLES_PER_CLASS]

dict = {i:[] for i in range(num_of_classes)}

counter = num_of_classes

# assembling all the indexes
for i,y in enumerate(y_train):
    if y >= num_of_classes:
        continue
    if len(dict[y]) < num_of_examples_per_class:
        dict[y].append(i)
        if len(dict[y]) == num_of_examples_per_class:
            counter -= 1
    if counter == 0:
        break

rows = num_of_classes
columns = num_of_examples_per_class

# plotting the collage
for title in ["Original", "Permutated"]:
    fig = plt.figure()
    fig.suptitle(title)
    for i in range(rows * columns):
        ax = fig.add_subplot(rows, columns, i + 1)
        _class = int(i / num_of_examples_per_class)
        _example = i % num_of_examples_per_class
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        img = x_train[dict[_class][_example]]
        img = p.pad(img, number_of_paddings=params[PADDING], padder=0.0)
        if title == "Permutated":
            img = e.encrypt(img)
        ax.imshow(img, cmap=plt.cm.binary)
        #ax.imshow(img)

plt.show()
