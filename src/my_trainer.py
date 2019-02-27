import src.my_Models as mdl
import numpy as np
import tensorflow as tf


data_types = {'fashion_mnist':tf.keras.datasets.fashion_mnist, 'mnist':tf.keras.datasets.mnist, 'cifar10':tf.keras.datasets.cifar10}
models = {"CW_1": mdl.CW_1, "CW_2": mdl.CW_2, "FGSM": mdl.FGSM}
train_mode = {"CTR":"my_ctr","CBC":"my_cbc","ECB":"my_ecb", "PERMUTATED":"my_permutated", "UNENCRYPTED":"my_unencrypted"}


def main():
    data = data_types[DATASET]

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    helper = __import__(train_mode[TRAIN_WITH_ME])

    dims = np.array(x_train).shape

    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    input_shape = np.array(x_train[0]).shape

    # getting the desired model
    model = models[MODEL](input_shape, encrypt=helper.encrypt)

    # training
    loss, epoch_accs, epochs = model.train(x_train, y_train, ep=5)

    # evaluating
    model.compile()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("{0} {1} {2}\n".format(DATASET, MODEL, test_acc))

    helper.print_results(test_acc, out=r, dataset=DATASET, model=MODEL,epoch_accuracies=epoch_accs,epochs=epochs)

    # saving model
    model.save(MODEL_NAME)


if __name__ == '__main__':
    # these two change to get desired model
    DATASET = "mnist"
    MODEL = "CW_1"
    TRAIN_WITH_ME = "UNENCRYPTED"

    MODEL_NAME = DATASET + "_" + MODEL + "_model"

    r = open("../results_temp", "a")
    main()
    r.close()