from Crypto.Util import Counter

import src.Models as mdl
import numpy as np
import tensorflow as tf
import json
import os # TODO urandom or random ?
from Crypto.Cipher import AES

data_types = {'fashion_mnist': tf.keras.datasets.fashion_mnist, 'mnist': tf.keras.datasets.mnist,
              'cifar10': tf.keras.datasets.cifar10}
models = {"CW_1": mdl.CW_1(), "CW_2": mdl.CW_2(), "FGSM": mdl.FGSM()}

def main():
    mode_of_operation = "CTR"

    data = data_types[DATASET]

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dims = np.array(x_train).shape

    seed = 42  # FOR DEBUGGING PERPUSES

    key = bytes(os.urandom(16))

    IV = Counter.new(16 * 8)

    aes_cipher = AES.new(key, AES.MODE_CTR, counter=IV)

    for image in x_train:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    x_test_aes = x_test.copy()

    for image in x_test_aes:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    print(x_test_aes.shape)

    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_test_aes = np.expand_dims(x_test_aes, -1)

    input_shape = np.array(x_train[0]).shape

    # getting the desired model
    model = models[MODEL]

    # building the networks' structure
    model.build(input_shape)

    # training
    loss, acc, epochs = model.train(x_train, y_train, ep=5)

    results = {
        "name": MODEL_NAME,
        "acc": acc,
        "epochs": epochs
    }

    # writing the training results
    with open("../AES_ " + mode_of_operation + "_encrypted_{}_models.json".format(DATASET), 'a') as j:
        json.dump(results, j)
        j.write('\n')

    # evaluating
    model.compile()

    test_loss, test_acc = model.evaluate(x_test_aes, y_test)
    print("{} {} {}\n".format(DATASET + " AES_" + mode_of_operation + "_model", MODEL, test_acc))
    r.write(
        "{} {} {} key is : {} , IV of counter is : {}\n".format(DATASET + " AES_" + mode_of_operation + "_model", MODEL,
                                                                test_acc, key, IV))

    # saving model
    model.save(MODEL_NAME)

    mode_of_operation = "ECB"

    data = data_types[DATASET]

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dims = np.array(x_train).shape

    seed = 42  # FOR DEBUGGING PERPUSES

    # key = 'abcdefghijklmnop' #TODO debug

    key = bytes(os.urandom(16))
    aes_cipher = AES.new(key, AES.MODE_ECB)

    for image in x_train:
        flattened = bytes(map(int,image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    x_test_aes = x_test.copy()

    for image in x_test_aes:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    print(x_test_aes.shape)

    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_test_aes = np.expand_dims(x_test_aes, -1)

    input_shape = np.array(x_train[0]).shape

    # getting the desired model
    model = models[MODEL]

    # building the networks' structure
    model.build(input_shape)

    # training
    loss, acc, epochs = model.train(x_train, y_train, ep=5)

    results = {
        "name": MODEL_NAME,
        "acc": acc,
        "epochs": epochs
    }

    # writing the training results
    with open("../AES_ " + mode_of_operation + "_encrypted_{}_models.json".format(DATASET), 'a') as j:
        json.dump(results, j)
        j.write('\n')

    # evaluating
    model.compile()

    test_loss, test_acc = model.evaluate(x_test_aes, y_test)
    print("{} {} {}\n".format(DATASET + " AES_" + mode_of_operation + "_model", MODEL, test_acc))
    r.write("{} {} {} key is {}\n".format(DATASET + " AES_" + mode_of_operation + "_model", MODEL, test_acc,key))

    # saving model
    model.save(MODEL_NAME)

    mode_of_operation = "CBC"

    data = data_types[DATASET]

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    dims = np.array(x_train).shape

    seed = 42  # FOR DEBUGGING PERPUSES

    key = bytes(os.urandom(16))

    IV = bytes(os.urandom(16))

    aes_cipher = AES.new(key, AES.MODE_CBC,IV=IV)

    for image in x_train:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    x_test_aes = x_test.copy()

    for image in x_test_aes:
        flattened = bytes(map(int, image.flatten().tolist()))
        aes_flattened = list(aes_cipher.encrypt(flattened))
        image[...] = np.reshape(aes_flattened, (dims[1], dims[2]))

    print(x_test_aes.shape)

    if len(dims) != 4:
        # expanding the images to get a third dimension (needed for conv layers)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_test_aes = np.expand_dims(x_test_aes, -1)

    input_shape = np.array(x_train[0]).shape

    # getting the desired model
    model = models[MODEL]

    # building the networks' structure
    model.build(input_shape)

    # training
    loss, acc, epochs = model.train(x_train, y_train, ep=5)

    results = {
        "name": MODEL_NAME,
        "acc": acc,
        "epochs": epochs
    }

    # writing the training results
    with open("../AES_ " + mode_of_operation + "_encrypted_{}_models.json".format(DATASET), 'a') as j:
        json.dump(results, j)
        j.write('\n')

    # evaluating
    model.compile()

    test_loss, test_acc = model.evaluate(x_test_aes, y_test)
    print("{} {} {}\n".format(DATASET + " AES_" + mode_of_operation + "_model", MODEL, test_acc))
    r.write("{} {} {} key is {} , IV is {}\n".format(DATASET + " AES_" + mode_of_operation + "_model", MODEL, test_acc, key,IV))

    # saving model
    model.save(MODEL_NAME)


def aes_encrypt(mode_of_operation, key):
    '''
    mode_of_operation can be one of the following: ECB,CBC,CTR . (AES.Mode...)
    '''

    cipher = AES.new(key, mode_of_operation)

    plaintext = [round(x / 5) for x in range(784)]
    plaintext = bytes(plaintext)

    msg = cipher.encrypt(plaintext)
    print(type(msg))
    print(msg)

    decipher = AES.new(key, AES.MODE_ECB)
    print(decipher.decrypt(msg))


if __name__ == '__main__':
    # these two change to get desired model
    DATASET = "mnist"
    MODEL = "CW_1"

    MODEL_NAME = DATASET + "_" + MODEL + "_aes_encrypted_model"

    r = open("../results1", "a")
    for dataset in ["mnist","fashion_mnist"]:
        for model in ["CW_1","CW_2","FGSM"]:
            DATASET = dataset
            MODEL = model
            main()
    r.close()
