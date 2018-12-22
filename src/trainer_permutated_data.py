import src.Models as mdl
import numpy as np
import tensorflow as tf

data_types = {'fashion_mnist': tf.keras.datasets.fashion_mnist, 'mnist': tf.keras.datasets.mnist,
              'cifar10': tf.keras.datasets.cifar10}
models = {"CW_1": mdl.CW_1(), "CW_2": mdl.CW_2(), "FGSM": mdl.FGSM()}

DATASET = "mnist"
MODEL = "CW_1"


def main():
    global DATASET, MODEL

    data = data_types[DATASET]

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0



    dims = np.array(x_train).shape

    seed = 42  # FOR DEBUGGING PERPUSES

    for image in x_train:
        permutated_flattened = np.random.RandomState(seed=seed).permutation(image.flatten())
        image[...] = np.reshape(permutated_flattened, (dims[1], dims[2]))

    x_test_permutated = x_test.copy()

    for image in x_test_permutated:
        permutated_flattened = np.random.RandomState(seed=seed).permutation(image.flatten())
        image[...] = np.reshape(permutated_flattened, (dims[1], dims[2]))

    print(x_test_permutated.shape)

    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_test_permutated = np.expand_dims(x_test_permutated, -1)

    input_shape = np.array(x_train[0]).shape

    # getting the desired model
    model = models[MODEL]

    # building the networks' structure
    model.build(input_shape)

    # training
    model.train(x_train, y_train, ep=5)

    # evaluating
    model.compile()



    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("{0} {1} {2}\n".format(DATASET, MODEL, test_acc))
    r.write("{0} {1} {2}\n".format(DATASET + " permutated_no_seed", MODEL, test_acc))

    test_perm_loss, test_perm_acc = model.evaluate(x_test_permutated, y_test)
    print("{0} {1} {2}\n".format(DATASET, MODEL, test_perm_acc))
    r.write("{0} {1} {2}\n".format(DATASET + " permutated_no_seed", MODEL + " permutated_test_no_seed", test_perm_acc))

    # saving model
    model.save(DATASET + "_" + MODEL + "permutated_model_no_seed")


if __name__ == '__main__':
    r = open("../results", "a")
    main()
    r.close()
